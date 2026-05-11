/**
 * @file spmc_workqueue.c
 * @brief Реализация блокирующейся (lock-free) SPMC очереди с поддержкой пакетов.
 */

#include "../include/t_master.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>

/* ============================================================================
 * Блокирующийся пул изображений (стек с tagged pointer)
 * ============================================================================ */

#define BLOCK_NEXT_OFFSET 0
#define TAG_BITS 8
#define INDEX_MASK ((UINTPTR_MAX) >> TAG_BITS)

typedef struct image_pool_block {
    uintptr_t next;
} image_pool_block_t;

struct image_pool {
    uint8_t* base;
    size_t block_size;
    uint32_t num_blocks;
    _Atomic uintptr_t freelist_head;
};

static inline uintptr_t block_index(image_pool_t *pool, uint8_t *block) {
    return (uintptr_t)((block - pool->base) / pool->block_size);
}

static inline uint8_t* block_from_index(image_pool_t *pool, uintptr_t idx) {
    return pool->base + idx * pool->block_size;
}

image_pool_t* image_pool_create(size_t block_size, uint32_t num_blocks) {
    if (block_size < MAX_IMAGE_SIZE || num_blocks == 0) return NULL;

    image_pool_t *pool = aligned_alloc(CACHE_LINE_SIZE, sizeof(image_pool_t));
    if (!pool) return NULL;

    pool->base = aligned_alloc(CACHE_LINE_SIZE, num_blocks * block_size);
    if (!pool->base) {
        free(pool);
        return NULL;
    }

    pool->block_size = block_size;
    pool->num_blocks = num_blocks;

    for (uint32_t i = 0; i < num_blocks; i++) {
        image_pool_block_t *blk = (image_pool_block_t*)(pool->base + i * block_size);
        blk->next = (i == num_blocks - 1) ? UINTPTR_MAX : (i + 1);
    }
    atomic_init(&pool->freelist_head, 0);
    return pool;
}

void* image_pool_alloc(image_pool_t *pool) {
    uintptr_t old_head, new_head;
    uint8_t *block;

    do {
        old_head = atomic_load_explicit(&pool->freelist_head, memory_order_acquire);
        if (old_head == UINTPTR_MAX) return NULL;

        uintptr_t idx = old_head & INDEX_MASK;
        block = block_from_index(pool, idx);
        uintptr_t next = *(uintptr_t*)(block + BLOCK_NEXT_OFFSET);

        uintptr_t counter = (old_head >> (8 * sizeof(uintptr_t) - TAG_BITS)) + 1;
        new_head = (next & INDEX_MASK) | (counter << (8 * sizeof(uintptr_t) - TAG_BITS));
        if (next == UINTPTR_MAX) new_head = UINTPTR_MAX;

    } while (!atomic_compare_exchange_weak_explicit(&pool->freelist_head,
                                                     &old_head, new_head,
                                                     memory_order_acq_rel,
                                                     memory_order_acquire));
    return block;
}

void image_pool_free(image_pool_t *pool, void *ptr) {
    if (!ptr) return;

    uint8_t *block = (uint8_t*)ptr;
    uintptr_t idx = block_index(pool, block);
    uintptr_t old_head, new_head;

    do {
        old_head = atomic_load_explicit(&pool->freelist_head, memory_order_acquire);
        uintptr_t next = (old_head == UINTPTR_MAX) ? UINTPTR_MAX : (old_head & INDEX_MASK);
        *(uintptr_t*)(block + BLOCK_NEXT_OFFSET) = next;

        uintptr_t counter = (old_head >> (8 * sizeof(uintptr_t) - TAG_BITS)) + 1;
        new_head = (idx & INDEX_MASK) | (counter << (8 * sizeof(uintptr_t) - TAG_BITS));
        if (old_head == UINTPTR_MAX) new_head = idx;

    } while (!atomic_compare_exchange_weak_explicit(&pool->freelist_head,
                                                     &old_head, new_head,
                                                     memory_order_release,
                                                     memory_order_relaxed));
}

void image_pool_destroy(image_pool_t *pool) {
    if (!pool) return;
    free(pool->base);
    free(pool);
}

/* ============================================================================
 * Вспомогательные функции для очереди
 * ============================================================================ */

static void update_min_read_cursor(spmc_workqueue_t *queue) {
    uint64_t min_cursor = UINT64_MAX;
    for (uint32_t i = 0; i < MAX_WORKERS; i++) {
        if (atomic_load_explicit(&queue->workers[i].active, memory_order_acquire)) {
            uint64_t cursor = atomic_load_explicit(&queue->workers[i].cursor,
                                                   memory_order_acquire);
            if (cursor < min_cursor) min_cursor = cursor;
        }
    }
    if (min_cursor != UINT64_MAX) {
        atomic_store_explicit(&queue->min_read_cursor, min_cursor, memory_order_release);
    }
}

/* ============================================================================
 * Управление очередью
 * ============================================================================ */

spmc_workqueue_t* workqueue_create(uint32_t capacity, uint32_t image_pool_blocks) {
    if (capacity == 0) capacity = DEFAULT_QUEUE_CAPACITY;
    if (!is_power_of_2(capacity) || capacity < 2) return NULL;

    if (image_pool_blocks == 0) image_pool_blocks = capacity * 2;

    spmc_workqueue_t *queue = aligned_alloc(CACHE_LINE_SIZE, sizeof(spmc_workqueue_t));
    if (!queue) return NULL;
    memset(queue, 0, sizeof(*queue));

    size_t slots_size = sizeof(work_slot_t) * capacity;
    queue->slots = aligned_alloc(CACHE_LINE_SIZE, slots_size);
    if (!queue->slots) {
        free(queue);
        return NULL;
    }
    memset(queue->slots, 0, slots_size);

    queue->image_pool = image_pool_create(MAX_IMAGE_SIZE, image_pool_blocks);
    if (!queue->image_pool) {
        free(queue->slots);
        free(queue);
        return NULL;
    }

    queue->capacity = capacity;
    queue->index_mask = capacity - 1;
    atomic_init(&queue->write_cursor, 0);
    atomic_init(&queue->min_read_cursor, 0);
    atomic_init(&queue->active_workers, 0);

    for (uint32_t i = 0; i < capacity; i++) {
        atomic_init(&queue->slots[i].sequence, i);
        atomic_init(&queue->slots[i].state, TASK_STATE_EMPTY);
        atomic_init(&queue->slots[i].worker_id, -1);
    }

    for (uint32_t i = 0; i < MAX_WORKERS; i++) {
        atomic_init(&queue->workers[i].cursor, 0);
        atomic_init(&queue->workers[i].active, false);
        queue->workers[i].worker_id = i;
        memset(&queue->workers[i].stats, 0, sizeof(worker_stats_t));
        queue->workers[i].stats.min_latency_ns = UINT64_MAX;
    }

    return queue;
}

void workqueue_destroy(spmc_workqueue_t *queue) {
    if (!queue) return;
    for (uint32_t i = 0; i < MAX_WORKERS; i++)
        atomic_store_explicit(&queue->workers[i].active, false, memory_order_release);
    if (queue->image_pool) image_pool_destroy(queue->image_pool);
    free(queue->slots);
    free(queue);
}

bool workqueue_register_worker(spmc_workqueue_t *queue, worker_handle_t *worker) {
    if (!queue || !worker) return false;
    for (uint32_t i = 0; i < MAX_WORKERS; i++) {
        bool expected = false;
        if (atomic_compare_exchange_strong_explicit(&queue->workers[i].active,
                                                     &expected, true,
                                                     memory_order_acq_rel,
                                                     memory_order_acquire)) {
            uint64_t current_write = atomic_load_explicit(&queue->write_cursor,
                                                          memory_order_acquire);
            worker->queue = queue;
            worker->worker_id = i;
            worker->local_cursor = current_write;
            atomic_store_explicit(&queue->workers[i].cursor, current_write,
                                 memory_order_release);
            atomic_fetch_add_explicit(&queue->active_workers, 1, memory_order_acq_rel);
            update_min_read_cursor(queue);
            return true;
        }
    }
    return false;
}

void workqueue_unregister_worker(worker_handle_t *worker) {
    if (!worker || !worker->queue) return;
    spmc_workqueue_t *queue = worker->queue;
    uint32_t id = worker->worker_id;
    if (id >= MAX_WORKERS) return;
    atomic_store_explicit(&queue->workers[id].active, false, memory_order_release);
    atomic_fetch_sub_explicit(&queue->active_workers, 1, memory_order_acq_rel);
    update_min_read_cursor(queue);
    worker->queue = NULL;
}

/* ============================================================================
 * Реализация производителя (одиночная постановка)
 * ============================================================================ */

static atomic_uint_fast64_t global_request_counter = 0;

bool workqueue_enqueue(spmc_workqueue_t *queue,
                       const void *image_data,
                       uint32_t image_size,
                       const client_info_t *client) {
    if (!queue || !image_data || !client || image_size == 0 || image_size > MAX_IMAGE_SIZE)
        return false;

    void *img_ptr = image_pool_alloc(queue->image_pool);
    if (!img_ptr) {
        queue->stats.queue_full_events++;
        return false;
    }
    memcpy(img_ptr, image_data, image_size);

    uint64_t write_pos = atomic_load_explicit(&queue->write_cursor, memory_order_relaxed);
    uint32_t index = write_pos & queue->index_mask;
    work_slot_t *slot = &queue->slots[index];

    uint64_t expected_seq = write_pos;
    uint32_t spin = 0;
    while (1) {
        uint64_t current_seq = atomic_load_explicit(&slot->sequence, memory_order_acquire);
        if (current_seq == expected_seq) {
            int state = atomic_load_explicit(&slot->state, memory_order_acquire);
            if (state == TASK_STATE_EMPTY || state == TASK_STATE_DONE)
                break;
        }
        if (write_pos >= current_seq + queue->capacity) {
            uint64_t min_read = atomic_load_explicit(&queue->min_read_cursor,
                                                     memory_order_acquire);
            if (write_pos >= min_read + queue->capacity) {
                image_pool_free(queue->image_pool, img_ptr);
                queue->stats.queue_full_events++;
                return false;
            }
            update_min_read_cursor(queue);
        }
        if (++spin > ENQUEUE_SPIN_LIMIT) {
            image_pool_free(queue->image_pool, img_ptr);
            return false;
        }
        cpu_relax(spin);
    }

    slot->task.request_id = atomic_fetch_add_explicit(&global_request_counter, 1,
                                                      memory_order_relaxed);
    slot->task.timestamp_ns = get_timestamp_ns();
    slot->task.image_size = image_size;
    slot->task.image_ptr = img_ptr;
    slot->task.client = *client;
    slot->task.slot_position = write_pos;

    atomic_store_explicit(&slot->worker_id, -1, memory_order_release);
    atomic_store_explicit(&slot->state, TASK_STATE_PENDING, memory_order_release);
    atomic_store_explicit(&slot->sequence, write_pos + 1, memory_order_release);
    atomic_store_explicit(&queue->write_cursor, write_pos + 1, memory_order_release);

    queue->stats.tasks_enqueued++;
    return true;
}

/* ============================================================================
 * Пакетная постановка
 * ============================================================================ */

size_t workqueue_enqueue_batch(spmc_workqueue_t *queue,
                               size_t count,
                               const void *images[],
                               const uint32_t sizes[],
                               const client_info_t clients[]) {
    if (!queue || count == 0 || count > BATCH_MAX) return 0;

    void *ptrs[BATCH_MAX];
    size_t successful = 0;
    for (size_t i = 0; i < count; i++) {
        if (!images[i] || !clients || sizes[i] == 0 || sizes[i] > MAX_IMAGE_SIZE) {
            for (size_t j = 0; j < successful; j++)
                image_pool_free(queue->image_pool, ptrs[j]);
            return 0;
        }
        void *p = image_pool_alloc(queue->image_pool);
        if (!p) {
            for (size_t j = 0; j < successful; j++)
                image_pool_free(queue->image_pool, ptrs[j]);
            queue->stats.queue_full_events++;
            return 0;
        }
        ptrs[i] = p;
        successful++;
    }

    uint64_t write_pos = atomic_load_explicit(&queue->write_cursor, memory_order_relaxed);
    uint32_t spin = 0;

    while (1) {
        uint64_t min_read = atomic_load_explicit(&queue->min_read_cursor, memory_order_acquire);
        if (write_pos + count <= min_read + queue->capacity)
            break;

        if (spin++ > ENQUEUE_SPIN_LIMIT) {
            for (size_t i = 0; i < count; i++)
                image_pool_free(queue->image_pool, ptrs[i]);
            queue->stats.queue_full_events++;
            return 0;
        }
        cpu_relax(spin);
        update_min_read_cursor(queue);
    }

    uint64_t new_write = write_pos + count;
    if (!atomic_compare_exchange_strong_explicit(&queue->write_cursor,
                                                  &write_pos, new_write,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        for (size_t i = 0; i < count; i++)
            image_pool_free(queue->image_pool, ptrs[i]);
        return 0;
    }

    for (size_t i = 0; i < count; i++) {
        uint32_t index = (write_pos + i) & queue->index_mask;
        work_slot_t *slot = &queue->slots[index];

        memcpy(ptrs[i], images[i], sizes[i]);

        slot->task.request_id = atomic_fetch_add_explicit(&global_request_counter, 1,
                                                          memory_order_relaxed);
        slot->task.timestamp_ns = get_timestamp_ns();
        slot->task.image_size = sizes[i];
        slot->task.image_ptr = ptrs[i];
        slot->task.client = clients[i];
        slot->task.slot_position = write_pos + i;

        atomic_store_explicit(&slot->worker_id, -1, memory_order_release);
        atomic_store_explicit(&slot->state, TASK_STATE_PENDING, memory_order_release);
        atomic_store_explicit(&slot->sequence, write_pos + i + 1, memory_order_release);
    }

    queue->stats.tasks_enqueued += count;
    return count;
}

/* ============================================================================
 * Реализация потребителя (одиночный захват)
 * ============================================================================ */

bool workqueue_try_claim(worker_handle_t *worker, task_descriptor_t *task) {
    if (!worker || !worker->queue || !task) return false;

    spmc_workqueue_t *queue = worker->queue;
    uint64_t read_pos = worker->local_cursor;
    uint32_t index = read_pos & queue->index_mask;
    work_slot_t *slot = &queue->slots[index];

    uint64_t expected_seq = read_pos + 1;
    uint64_t current_seq = atomic_load_explicit(&slot->sequence, memory_order_acquire);
    if (current_seq != expected_seq) return false;

    int expected_state = TASK_STATE_PENDING;
    if (!atomic_compare_exchange_strong_explicit(&slot->state,
                                                  &expected_state,
                                                  TASK_STATE_CLAIMED,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        queue->workers[worker->worker_id].stats.claim_conflicts++;
        worker->local_cursor = read_pos + 1;
        atomic_store_explicit(&queue->workers[worker->worker_id].cursor,
                             read_pos + 1, memory_order_release);
        return false;
    }

    atomic_store_explicit(&slot->worker_id, (int)worker->worker_id, memory_order_release);
    atomic_store_explicit(&slot->state, TASK_STATE_PROCESSING, memory_order_release);
    memcpy(task, &slot->task, sizeof(task_descriptor_t));

    worker->local_cursor = read_pos + 1;
    atomic_store_explicit(&queue->workers[worker->worker_id].cursor,
                         read_pos + 1, memory_order_release);

    queue->workers[worker->worker_id].stats.tasks_claimed++;
    queue->stats.tasks_dequeued++;

    if ((read_pos + 1) % 64 == 0)
        update_min_read_cursor(queue);

    return true;
}

/* ============================================================================
 * Пакетный захват
 * ============================================================================ */

size_t workqueue_try_claim_batch(worker_handle_t *worker,
                                 size_t max_count,
                                 task_descriptor_t tasks[]) {
    if (!worker || !worker->queue || max_count == 0 || max_count > BATCH_MAX || !tasks)
        return 0;

    spmc_workqueue_t *queue = worker->queue;
    uint64_t start_pos = worker->local_cursor;

    size_t avail = 0;
    for (size_t i = 0; i < max_count; i++) {
        uint32_t index = (start_pos + i) & queue->index_mask;
        work_slot_t *slot = &queue->slots[index];
        uint64_t expected_seq = start_pos + i + 1;
        uint64_t current_seq = atomic_load_explicit(&slot->sequence, memory_order_acquire);
        if (current_seq != expected_seq) break;
        int state = atomic_load_explicit(&slot->state, memory_order_acquire);
        if (state != TASK_STATE_PENDING) break;
        avail++;
    }

    if (avail == 0) return 0;

    uint64_t new_cursor = start_pos + avail;
    if (!atomic_compare_exchange_strong_explicit(&queue->workers[worker->worker_id].cursor,
                                                  &worker->local_cursor, new_cursor,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        return 0;
    }

    size_t claimed = 0;
    for (size_t i = 0; i < avail; i++) {
        uint32_t index = (start_pos + i) & queue->index_mask;
        work_slot_t *slot = &queue->slots[index];

        int expected_state = TASK_STATE_PENDING;
        if (atomic_compare_exchange_strong_explicit(&slot->state,
                                                     &expected_state,
                                                     TASK_STATE_CLAIMED,
                                                     memory_order_acq_rel,
                                                     memory_order_acquire)) {
            atomic_store_explicit(&slot->worker_id, (int)worker->worker_id,
                                 memory_order_release);
            atomic_store_explicit(&slot->state, TASK_STATE_PROCESSING,
                                 memory_order_release);
            memcpy(&tasks[claimed], &slot->task, sizeof(task_descriptor_t));
            tasks[claimed].slot_position = start_pos + i;
            claimed++;
        } else {
            queue->workers[worker->worker_id].stats.claim_conflicts++;
        }
    }

    if (claimed > 0) {
        worker->local_cursor = start_pos + claimed;
        atomic_store_explicit(&queue->workers[worker->worker_id].cursor,
                             start_pos + claimed, memory_order_release);
        queue->workers[worker->worker_id].stats.tasks_claimed += claimed;
        queue->stats.tasks_dequeued += claimed;
    } else {
        worker->local_cursor = start_pos;
        atomic_store_explicit(&queue->workers[worker->worker_id].cursor,
                             start_pos, memory_order_release);
    }

    if ((start_pos + claimed) % 64 == 0)
        update_min_read_cursor(queue);

    return claimed;
}

/* ============================================================================
 * Завершение задачи
 * ============================================================================ */

void workqueue_complete(worker_handle_t *worker,
                       const task_descriptor_t *task,
                       bool success) {
    if (!worker || !worker->queue || !task) return;

    spmc_workqueue_t *queue = worker->queue;
    uint64_t now = get_timestamp_ns();
    uint64_t latency = now - task->timestamp_ns;
    worker_stats_t *stats = &queue->workers[worker->worker_id].stats;

    if (success) {
        stats->tasks_completed++;
        stats->total_processing_ns += latency;
        if (latency < stats->min_latency_ns) stats->min_latency_ns = latency;
        if (latency > stats->max_latency_ns) stats->max_latency_ns = latency;
    } else {
        stats->tasks_failed++;
    }

    image_pool_free(queue->image_pool, task->image_ptr);

    uint32_t index = task->slot_position & queue->index_mask;
    work_slot_t *slot = &queue->slots[index];
    atomic_store_explicit(&slot->state, TASK_STATE_DONE, memory_order_release);
}

/* ============================================================================
 * Статистика
 * ============================================================================ */

void workqueue_get_worker_stats(const worker_handle_t *worker, worker_stats_t *stats) {
    if (!worker || !worker->queue || worker->worker_id >= MAX_WORKERS || !stats) return;
    memcpy(stats, &worker->queue->workers[worker->worker_id].stats, sizeof(worker_stats_t));
}

void workqueue_get_queue_stats(const spmc_workqueue_t *queue, queue_stats_t *stats) {
    if (!queue || !stats) return;
    memcpy(stats, &queue->stats, sizeof(queue_stats_t));
}

uint32_t workqueue_depth(const spmc_workqueue_t *queue) {
    if (!queue) return 0;
    uint64_t write = atomic_load_explicit(&queue->write_cursor, memory_order_acquire);
    uint64_t min_read = atomic_load_explicit(&queue->min_read_cursor, memory_order_acquire);
    return (uint32_t)(write - min_read);
}

bool workqueue_is_empty(const spmc_workqueue_t *queue) {
    return workqueue_depth(queue) == 0;
}

bool workqueue_is_full(const spmc_workqueue_t *queue) {
    if (!queue) return true;
    uint64_t write = atomic_load_explicit(&queue->write_cursor, memory_order_acquire);
    uint64_t min_read = atomic_load_explicit(&queue->min_read_cursor, memory_order_acquire);
    return (write - min_read) >= queue->capacity;
}
