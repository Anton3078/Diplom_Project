/**
 * @file t_master.h
 * @brief Блокирующаяся (lock-free) SPMC очередь задач с поддержкой пакетной обработки.
 */

#ifndef SPMC_WORKQUEUE_H
#define SPMC_WORKQUEUE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <time.h>
#include <sched.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Конфигурационные константы
 * ============================================================================ */

#define CACHE_LINE_SIZE 64
#define MAX_IMAGE_SIZE   (2 * 1024 * 1024)
#define DEFAULT_QUEUE_CAPACITY 256
#define MAX_WORKERS      2
#define MAX_CLIENTS      1024
#define BATCH_MAX        32

#define TASK_STATE_EMPTY      0
#define TASK_STATE_PENDING    1
#define TASK_STATE_CLAIMED    2
#define TASK_STATE_PROCESSING 3
#define TASK_STATE_DONE       4

#ifndef ENQUEUE_SPIN_LIMIT
#define ENQUEUE_SPIN_LIMIT 10000
#endif

#define MAX_FACES_PER_IMAGE 32
#define EMBEDDING_SIZE      512    // может не использоваться, но оставлено для совместимости

#define ERR_DECODE          1
#define ERR_MODEL_DETECTION 2
#define ERR_MODEL_RECOGNITION 3
#define ERR_INTERNAL        4

/* ============================================================================
 * Структуры сообщений
 * ============================================================================ */

struct error_info {
    uint8_t type : 4;
    uint8_t code : 4;
} __attribute__((packed));

typedef struct {
    int socket_fd;
    struct sockaddr_storage addr;
    socklen_t addr_len;
    uint64_t client_id;
} client_info_t;

/**
 * @brief Результат по одному лицу.
 */
typedef struct {
    float x1, y1, x2, y2;          // абсолютные координаты
    float confidence;                // уверенность детектора
    uint32_t class_id;                // идентификатор класса (от 0 до N-1)
    float embedding[EMBEDDING_SIZE]; // опционально
} face_result_t;

/**
 * @brief Итоговый результат задачи (отправляется клиенту).
 */
typedef struct {
    int face_count;
    face_result_t faces[MAX_FACES_PER_IMAGE];
    struct error_info err;
} task_result_t;

/**
 * @brief Дескриптор задачи (хранится в слоте очереди).
 */
typedef struct {
    uint64_t request_id;
    uint64_t timestamp_ns;
    uint32_t image_size;
    void    *image_ptr;
    client_info_t client;
    uint64_t slot_position;
} task_descriptor_t;

/**
 * @brief Слот очереди.
 */
typedef struct {
    alignas(CACHE_LINE_SIZE) atomic_uint_fast64_t sequence;
    atomic_int state;
    atomic_int worker_id;
    task_descriptor_t task;
    uint8_t _padding[CACHE_LINE_SIZE - ((sizeof(atomic_uint_fast64_t) +
                                         sizeof(atomic_int) * 2 +
                                         sizeof(task_descriptor_t)) % CACHE_LINE_SIZE)];
} work_slot_t;

/**
 * @brief Статистика рабочего потока.
 */
typedef struct {
    uint64_t tasks_claimed;
    uint64_t tasks_completed;
    uint64_t tasks_failed;
    uint64_t claim_conflicts;
    uint64_t total_processing_ns;
    uint64_t min_latency_ns;
    uint64_t max_latency_ns;
} worker_stats_t;

/**
 * @brief Контекст рабочего потока.
 */
typedef struct {
    alignas(CACHE_LINE_SIZE) atomic_uint_fast64_t cursor;
    atomic_bool active;
    uint32_t worker_id;
    worker_stats_t stats;
    uint32_t _padding[10];
} worker_context_t;

/**
 * @brief Статистика очереди.
 */
typedef struct {
    uint64_t tasks_enqueued;
    uint64_t tasks_dequeued;
    uint64_t queue_full_events;
    uint64_t avg_queue_depth;
} queue_stats_t;

/**
 * @brief Основная структура SPMC очереди.
 */
typedef struct {
    alignas(CACHE_LINE_SIZE) atomic_uint_fast64_t write_cursor;
    uint32_t capacity;
    uint32_t index_mask;
    uint32_t _padding0[13];

    alignas(CACHE_LINE_SIZE) atomic_uint_fast64_t min_read_cursor;
    atomic_uint_fast32_t active_workers;
    uint32_t _padding1[14];

    alignas(CACHE_LINE_SIZE) worker_context_t workers[MAX_WORKERS];

    alignas(CACHE_LINE_SIZE) queue_stats_t stats;

    work_slot_t *slots;
    void *image_pool;
} spmc_workqueue_t;

/**
 * @brief Дескриптор рабочего потока.
 */
typedef struct {
    spmc_workqueue_t *queue;
    uint32_t worker_id;
    uint64_t local_cursor;
} worker_handle_t;

/* ============================================================================
 * Конфигурация рабочего потока
 * ============================================================================ */

typedef struct worker_config {
    spmc_workqueue_t* queue;                // Очередь задач
    const char* detection_model_path;       // Путь к модели детекции
    const char* recognition_model_path;      // Путь к модели распознавания
} worker_config_t;

/* ============================================================================
 * Вспомогательные функции
 * ============================================================================ */

static inline bool is_power_of_2(uint32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline void cpu_relax(uint32_t iteration) {
    if (iteration < 10) {
        for (volatile uint32_t i = 0; i < (1u << iteration); i++) {
            __asm__ volatile("pause" ::: "memory");
        }
    } else if (iteration < 20) {
        sched_yield();
    } else {
        struct timespec ts = {0, 1000};
        nanosleep(&ts, NULL);
    }
}

/* ============================================================================
 * Интерфейс пула изображений
 * ============================================================================ */

typedef struct image_pool image_pool_t;

image_pool_t* image_pool_create(size_t block_size, uint32_t num_blocks);
void* image_pool_alloc(image_pool_t *pool);
void image_pool_free(image_pool_t *pool, void *ptr);
void image_pool_destroy(image_pool_t *pool);

/* ============================================================================
 * Управление очередью
 * ============================================================================ */

spmc_workqueue_t* workqueue_create(uint32_t capacity, uint32_t image_pool_blocks);
void workqueue_destroy(spmc_workqueue_t *queue);
bool workqueue_register_worker(spmc_workqueue_t *queue, worker_handle_t *worker);
void workqueue_unregister_worker(worker_handle_t *worker);

/* ============================================================================
 * Интерфейс производителя
 * ============================================================================ */

bool workqueue_enqueue(spmc_workqueue_t *queue,
                       const void *image_data,
                       uint32_t image_size,
                       const client_info_t *client);

size_t workqueue_enqueue_batch(spmc_workqueue_t *queue,
                               size_t count,
                               const void *images[],
                               const uint32_t sizes[],
                               const client_info_t clients[]);

/* ============================================================================
 * Интерфейс потребителя
 * ============================================================================ */

bool workqueue_try_claim(worker_handle_t *worker, task_descriptor_t *task);
size_t workqueue_try_claim_batch(worker_handle_t *worker,
                                 size_t max_count,
                                 task_descriptor_t tasks[]);
void workqueue_complete(worker_handle_t *worker,
                       const task_descriptor_t *task,
                       bool success);

/* ============================================================================
 * Статистика
 * ============================================================================ */

void workqueue_get_worker_stats(const worker_handle_t *worker, worker_stats_t *stats);
void workqueue_get_queue_stats(const spmc_workqueue_t *queue, queue_stats_t *stats);
uint32_t workqueue_depth(const spmc_workqueue_t *queue);
bool workqueue_is_empty(const spmc_workqueue_t *queue);
bool workqueue_is_full(const spmc_workqueue_t *queue);

#ifdef __cplusplus
}
#endif

#endif /* SPMC_WORKQUEUE_H */
