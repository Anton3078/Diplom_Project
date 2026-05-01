/**
 * @file worker_thread.c
 * @brief Реализация рабочего потока с каскадной обработкой Detection + Recognition.
 */

#include "../../include/t_master.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>

// Библиотеки для декодирования
#include <turbojpeg.h>
#include <spng.h>

// Библиотеки для ресайза
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../include/stb_image_resize2.h"

// ONNX Runtime
#include <onnxruntime_c_api.h>

#define DET_MODEL_WIDTH  224
#define DET_MODEL_HEIGHT 224
#define REC_MODEL_WIDTH  160
#define REC_MODEL_HEIGHT 160
#define NUM_CLASSES      100

// Локальный контекст рабочего потока
typedef struct worker_thread_context {
    const OrtApi* ort;
    OrtEnv* env;
    OrtSession* detection_session;
    OrtSession* recognition_session;
    OrtMemoryInfo* memory_info;
    float det_mean[3];
    float det_std[3];
    float rec_mean[3];
    float rec_std[3];
    float* know_embeddings;
    int* know_labels;
    int num_persons;
    const char* det_input_names[1];
    const char* det_output_names[1];
    const char* rec_input_names[1];
    const char* rec_output_names[1];
} worker_thread_context_t;

static volatile sig_atomic_t worker_threads_running = 1;

/* ============================================================================
 * Декодирование изображений
 * ============================================================================ */

static unsigned char* decode_jpeg(const uint8_t* buf, size_t len, int* w, int* h) {
    tjhandle tj = tjInitDecompress();
    if (!tj) return NULL;

    int subsamp, color_space;
    if (tjDecompressHeader3(tj, buf, len, w, h, &subsamp, &color_space) < 0) {
        tjDestroy(tj);
        return NULL;
    }

    unsigned char* pixels = malloc((*w) * (*h) * 3);
    if (!pixels) {
        tjDestroy(tj);
        return NULL;
    }

    if (tjDecompress2(tj, buf, len, pixels, *w, 0, *h, TJPF_RGB, TJFLAG_FASTDCT) < 0) {
        free(pixels);
        tjDestroy(tj);
        return NULL;
    }

    tjDestroy(tj);
    return pixels;
}

static unsigned char* decode_png(const uint8_t* buf, size_t len, int* w, int* h) {
    spng_ctx* ctx = spng_ctx_new(0);
    if (!ctx) return NULL;

    spng_set_png_buffer(ctx, buf, len);
    struct spng_ihdr ihdr;
    if (spng_get_ihdr(ctx, &ihdr) != 0) {
        spng_ctx_free(ctx);
        return NULL;
    }
    *w = ihdr.width;
    *h = ihdr.height;

    size_t out_size;
    if (spng_decoded_image_size(ctx, SPNG_FMT_RGB8, &out_size) != 0) {
        spng_ctx_free(ctx);
        return NULL;
    }

    unsigned char* pixels = malloc(out_size);
    if (!pixels) {
        spng_ctx_free(ctx);
        return NULL;
    }

    if (spng_decode_image(ctx, pixels, out_size, SPNG_FMT_RGB8, 0) != 0) {
        free(pixels);
        spng_ctx_free(ctx);
        return NULL;
    }

    spng_ctx_free(ctx);
    return pixels;
}

static unsigned char* decode_image(const uint8_t* buf, size_t len, int* w, int* h) {
    if (len >= 2 && buf[0] == 0xFF && buf[1] == 0xD8)
        return decode_jpeg(buf, len, w, h);
    if (len >= 4 && buf[0] == 0x89 && buf[1] == 0x50 && buf[2] == 0x4E && buf[3] == 0x47)
        return decode_png(buf, len, w, h);
    return NULL;
}

/* ============================================================================
 * Предобработка изображения
 * ============================================================================ */

static float* preprocess_rgb(const unsigned char* rgb, int src_w, int src_h,
                             int dst_w, int dst_h,
                             const float mean[3], const float std[3]) {
    unsigned char* resized = malloc(dst_w * dst_h * 3);
    if (!resized) return NULL;

    // Используем STBIR_RGB
    if (!stbir_resize_uint8_linear(rgb, src_w, src_h, 0,
                                   resized, dst_w, dst_h, 0,
                                   STBIR_RGB)) {
        free(resized);
        return NULL;
    }

    float* tensor = malloc(dst_w * dst_h * 3 * sizeof(float));
    if (!tensor) {
        free(resized);
        return NULL;
    }

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < dst_h; h++) {
            for (int w = 0; w < dst_w; w++) {
                int src_idx = h * dst_w * 3 + w * 3 + c;
                int dst_idx = c * dst_h * dst_w + h * dst_w + w;
                float val = resized[src_idx] / 255.0f;
                tensor[dst_idx] = (val - mean[c]) / std[c];
            }
        }
    }

    free(resized);
    return tensor;
}

/* ============================================================================
 * ONNX Runtime
 * ============================================================================ */

static void check_ort_status(const OrtApi* ort, OrtStatus* status, const char* msg) {
    if (status) {
        const char* err = ort->GetErrorMessage(status);
        fprintf(stderr, "ONNX Runtime error: %s: %s\n", msg, err);
        ort->ReleaseStatus(status);
        exit(EXIT_FAILURE);
    }
}

static OrtSession* create_session(const OrtApi* ort, OrtEnv* env, const char* model_path,
                                   OrtMemoryInfo** memory_info) {
    OrtSessionOptions* session_options;
    OrtStatus* status = ort->CreateSessionOptions(&session_options);
    check_ort_status(ort, status, "CreateSessionOptions");

    status = ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
    check_ort_status(ort, status, "SetSessionGraphOptimizationLevel");

    OrtSession* session;
    status = ort->CreateSession(env, model_path, session_options, &session);
    check_ort_status(ort, status, "CreateSession");
    ort->ReleaseSessionOptions(session_options);

    status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memory_info);
    check_ort_status(ort, status, "CreateCpuMemoryInfo");

    return session;
}

static int run_detection(OrtSession* session, const OrtApi* ort, OrtMemoryInfo* memory_info,
                         const float* tensor, int det_w, int det_h,
                         face_result_t* faces, int max_faces, uint32_t req_id) {
    if (max_faces < 1) return 0;

    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;

    int64_t input_shape[] = {1, 3, det_h, det_w};
    size_t input_size = 1 * 3 * det_h * det_w * sizeof(float);

    OrtStatus* status = ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)tensor,
                                                           input_size, input_shape, 4,
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor);
    if (status) {
        ort->ReleaseStatus(status);
        return -1;
    }

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    status = ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor,
                      1, output_names, 1, &output_tensor);
    if (status) {
        fprintf(stderr, "[Worker] Req %u: Inference FAILED. %s\n", req_id, ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        return -1;
    }
    
    float* output_data;
    status = ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        ort->ReleaseValue(output_tensor);
        return -1;
    }
    fprintf(stderr, "[Worker] Req %u: Raw model output[0..4] = %.4f, %.4f, %.4f, %.4f, %.4f\n", req_id, output_data[0], output_data[1], output_data[2], output_data[3], output_data[4]);

    float conf = output_data[0];
    if (conf > 0.5f) {
        faces[0].confidence = conf;
        faces[0].x1 = output_data[1];
        faces[0].y1 = output_data[2];
        faces[0].x2 = output_data[3];
        faces[0].y2 = output_data[4];
    } else {
        fprintf(stderr, "[Worker] Req %u: NO FACE. Confidence %.3f < 0.5\n", req_id, conf);
    }

    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);

    return (conf > 0.5f) ? 1 : 0;
}

static int run_recognition_embedding(OrtSession* session, const OrtApi* ort, OrtMemoryInfo* memory_info,
                           const float* tensor, int rec_w, int rec_h,
                           int* out_class, float* out_conf, worker_thread_context_t* ctx) {
    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;
    int64_t input_shape[] = {1, 3, rec_h, rec_w};
    size_t input_size = 1 * 3 * rec_h * rec_w * sizeof(float);

    OrtStatus* status = ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)tensor,
                                                           input_size, input_shape, 4,
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor);
    if (status) {
        ort->ReleaseStatus(status);
        return -1;
    }

    const char* input_names[] = {"input"};
    const char* output_names[] = {"embedding"};

    status = ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor,
                      1, output_names, 1, &output_tensor);
    if (status) {
        fprintf(stderr, "[Worker] Recognition embedding inference failed: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        return -1;
    }

    float* output_data;
    status = ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        ort->ReleaseValue(output_tensor);
        return -1;
    }
    
    float norm = 0.0f;
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        norm += output_data[i] * output_data[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f)
    {
        for (int i = 0; i < EMBEDDING_SIZE; i++) 
        {
            output_data[i] /= norm;
        }
    }

    float best_sim = -1.0f;
    int best_label = -1;
    for (int p = 0; p < ctx->num_persons; p++)
    {
        const float* know_emb = ctx->know_embeddings + p * EMBEDDING_SIZE;
        float dot = 0.0f;
        for (int i = 0; i < EMBEDDING_SIZE; i++) 
        {
            dot += output_data[i] * know_emb[i];
        }
        if (dot > best_sim)
        {
            best_sim = dot;
            best_label = ctx->know_labels[p];
        }
    }
    
    *out_class = best_label;
    *out_conf = best_sim;

    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);
    return 0;
}

/* ============================================================================
 * Контекст рабочего потока
 * ============================================================================ */

static worker_thread_context_t* worker_context_create(const worker_config_t* cfg) {
    worker_thread_context_t* ctx = calloc(1, sizeof(worker_thread_context_t));
    if (!ctx) return NULL;

    ctx->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ctx->ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        free(ctx);
        return NULL;
    }

    OrtStatus* status = ctx->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "face_server", &ctx->env);
    if (status) {
        const char* err = ctx->ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create ONNX env: %s\n", err);
        ctx->ort->ReleaseStatus(status);
        free(ctx);
        return NULL;
    }

    ctx->detection_session = create_session(ctx->ort, ctx->env, cfg->detection_model_path,
                                            &ctx->memory_info);
    ctx->recognition_session = create_session(ctx->ort, ctx->env, cfg->recognition_model_path,
                                              &ctx->memory_info);
    
    FILE* femb = fopen("./models/embeddings.bin", "rb");
    if (femb)
    {
        fseek(femb, 0, SEEK_END);
        long emb_bytes = ftell(femb);
        fseek(femb, 0, SEEK_SET);
        ctx->num_persons = emb_bytes / (sizeof(float) * EMBEDDING_SIZE);
        ctx->know_embeddings = malloc(emb_bytes);
        fread(ctx->know_embeddings, 1, emb_bytes, femb);
        fclose(femb);
        printf("[Worker] Loaded %d embeddings\n", ctx->num_persons);

        FILE* flbl = fopen("./models/labels.bin", "rb");

        if (flbl) 
        {
            ctx->know_labels = malloc(ctx->num_persons * sizeof(int));
            fread(ctx->know_labels,  sizeof(int), ctx->num_persons, flbl);
            fclose(flbl);
        }
    }

    ctx->det_mean[0] = 0.485f; ctx->det_mean[1] = 0.456f; ctx->det_mean[2] = 0.406f;
    ctx->det_std[0]  = 0.229f; ctx->det_std[1]  = 0.224f; ctx->det_std[2]  = 0.225f;
    ctx->rec_mean[0] = 0.5f; ctx->rec_mean[1] = 0.5f; ctx->rec_mean[2] = 0.5f;
    ctx->rec_std[0]  = 0.5f; ctx->rec_std[1]  = 0.5f; ctx->rec_std[2]  = 0.5f;

    ctx->det_input_names[0] = "input";
    ctx->det_output_names[0] = "output";
    ctx->rec_input_names[0] = "input";
    ctx->rec_output_names[0] = "output";

    return ctx;
}

static void worker_context_free(worker_thread_context_t* ctx) {
    if (!ctx) return;
    if (ctx->recognition_session) ctx->ort->ReleaseSession(ctx->recognition_session);
    if (ctx->detection_session) ctx->ort->ReleaseSession(ctx->detection_session);
    if (ctx->memory_info) ctx->ort->ReleaseMemoryInfo(ctx->memory_info);
    if (ctx->env) ctx->ort->ReleaseEnv(ctx->env);
    if (ctx->know_embeddings) { free(ctx->know_embeddings); ctx->know_embeddings = NULL; }
    if (ctx->know_labels) { free(ctx->know_labels); ctx->know_labels = NULL; }
    free(ctx);
}

/* ============================================================================
 * Отправка результата клиенту
 * ============================================================================ */

static bool send_result_to_client(int client_fd, const task_result_t* result) {
    if (client_fd < 0) return false;

    struct __attribute__((packed)) {
        int face_count;
        struct {
            float x1, y1, x2, y2;
            float confidence;
            uint32_t class_id;
            float embedding[EMBEDDING_SIZE];
        } faces[MAX_FACES_PER_IMAGE];
        struct error_info err;
    } response;
    

    response.face_count = result->face_count;
    for (int i = 0; i < result->face_count; i++) {
        response.faces[i].x1 = result->faces[i].x1;
        response.faces[i].y1 = result->faces[i].y1;
        response.faces[i].x2 = result->faces[i].x2;
        response.faces[i].y2 = result->faces[i].y2;
        response.faces[i].confidence = result->faces[i].confidence;
        response.faces[i].class_id = result->faces[i].class_id;
        memcpy(response.faces[i].embedding, result->faces[i].embedding, sizeof(float) * EMBEDDING_SIZE);
    }
    response.err = result->err;

    ssize_t sent = send(client_fd, &response, sizeof(response), MSG_NOSIGNAL);
    fprintf(stderr, "[Worker] Sent result to FD %d: %d faces, bytes_sent=%zd\n", client_fd, result->face_count, sent);
    return sent == sizeof(response);
}

/* ============================================================================
 * Обработка одной задачи
 * ============================================================================ */

static void process_task(task_descriptor_t* task, worker_thread_context_t* ctx) {
    task_result_t result = {0};
    result.face_count = 0;
    result.err.type = 0;
    result.err.code = 0;
    
    fprintf(stderr, "[Worker] Req %lu: Starting task. Raw size=%u\n", task->request_id, task->image_size);

    int img_w, img_h;
    unsigned char* rgb = decode_image(task->image_ptr, task->image_size, &img_w, &img_h);
    if (!rgb) {
        result.err.type = ERR_DECODE;
        result.err.code = 1;
        fprintf(stderr, "[Worker] Req %lu: DECODE FAILED\n", task->request_id);
        send_result_to_client(task->client.socket_fd, &result);
        return;
    }
    fprintf(stderr, "[Worker] Req %lu: Decoded OK. %dx%d. Resizing to %dx%d for model...\n",
            task->request_id, img_w, img_h, DET_MODEL_WIDTH, DET_MODEL_HEIGHT);

    float* det_tensor = preprocess_rgb(rgb, img_w, img_h,
                                       DET_MODEL_WIDTH, DET_MODEL_HEIGHT,
                                       ctx->det_mean, ctx->det_std);
    if (!det_tensor) {
        free(rgb);
        result.err.type = ERR_INTERNAL;
        result.err.code = 1;
        send_result_to_client(task->client.socket_fd, &result);
        return;
    }

    face_result_t det_faces[MAX_FACES_PER_IMAGE];
    int face_count = run_detection(ctx->detection_session, ctx->ort, ctx->memory_info,
                                   det_tensor, DET_MODEL_WIDTH, DET_MODEL_HEIGHT,
                                   det_faces, MAX_FACES_PER_IMAGE, task->request_id);
    free(det_tensor);

    if (face_count < 0) {
        free(rgb);
        result.err.type = ERR_MODEL_DETECTION;
        result.err.code = 1;
        send_result_to_client(task->client.socket_fd, &result);
        return;
    }

    for (int i = 0; i < face_count && i < MAX_FACES_PER_IMAGE; i++) {
        int x1 = (int)(det_faces[i].x1 * DET_MODEL_WIDTH);
        int y1 = (int)(det_faces[i].y1 * DET_MODEL_HEIGHT);
        int x2 = (int)(det_faces[i].x2 * DET_MODEL_WIDTH);
        int y2 = (int)(det_faces[i].y2 * DET_MODEL_HEIGHT);

        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 >= img_w) x2 = img_w - 1;
        if (y2 >= img_h) y2 = img_h - 1;
        int face_w = x2 - x1 + 1;
        int face_h = y2 - y1 + 1;
        if (face_w <= 0 || face_h <= 0) continue;

        unsigned char* face_rgb = malloc(face_w * face_h * 3);
        if (!face_rgb) continue;
        for (int fy = 0; fy < face_h; fy++) {
            memcpy(face_rgb + fy * face_w * 3,
                   rgb + (y1 + fy) * img_w * 3 + x1 * 3,
                   face_w * 3);
        }

        float* rec_tensor = preprocess_rgb(rgb, img_w, img_h,
                                           REC_MODEL_WIDTH, REC_MODEL_HEIGHT,
                                           ctx->rec_mean, ctx->rec_std);
        free(face_rgb);
        if (!rec_tensor) continue;

        int predicted_class = -1;
        float recog_conf = 0.0f;
        if (run_recognition_embedding(ctx->recognition_session, ctx->ort, ctx->memory_info,
                            rec_tensor, REC_MODEL_WIDTH, REC_MODEL_HEIGHT,
                            &predicted_class, &recog_conf, ctx) == 0) {
            result.faces[result.face_count].x1 = det_faces[i].x1 * img_w;
            result.faces[result.face_count].y1 = det_faces[i].y1 * img_h;
            result.faces[result.face_count].x2 = det_faces[i].x2 * img_w;
            result.faces[result.face_count].y2 = det_faces[i].y2 * img_h;
            result.faces[result.face_count].confidence = det_faces[i].confidence;
            result.faces[result.face_count].class_id = predicted_class;
            memset(result.faces[result.face_count].embedding, 0, sizeof(float) * EMBEDDING_SIZE);
            result.face_count++;
        }
        free(rec_tensor);
    }

    free(rgb);
    fprintf(stderr, "[Worker] Req %lu: Final face_count=%d. Sending to client FD %d...\n", task->request_id, result.face_count, task->client.socket_fd);
    send_result_to_client(task->client.socket_fd, &result);
    close(task->client.socket_fd);
}

/* ============================================================================
 * Основная функция потока
 * ============================================================================ */

void* worker_thread(void* arg) {
    worker_config_t* config = (worker_config_t*)arg;
    if (!config || !config->queue) {
        fprintf(stderr, "Ошибка: не передана очередь\n");
        return NULL;
    }

    worker_thread_context_t* ctx = worker_context_create(config);
    if (!ctx) {
        fprintf(stderr, "Ошибка инициализации контекста\n");
        return NULL;
    }

    worker_handle_t worker;
    if (!workqueue_register_worker(config->queue, &worker)) {
        fprintf(stderr, "Ошибка регистрации рабочего\n");
        worker_context_free(ctx);
        return NULL;
    }

    task_descriptor_t tasks[BATCH_MAX];
    fprintf(stderr, "[Worker ID %u] Registered to queue. Waiting for tasks...\n", worker.worker_id);

    while (worker_threads_running) {
        size_t claimed = workqueue_try_claim_batch(&worker, BATCH_MAX, tasks);
        if (claimed > 0) {
            fprintf(stderr, "[Worker ID %u] Claimed %zu tasks.\n", worker.worker_id, claimed);
            for (size_t i = 0; i < claimed; i++) {
                process_task(&tasks[i], ctx);
                workqueue_complete(&worker, &tasks[i], true);
            }
        } else {
            cpu_relax(10);
        }
    }

    workqueue_unregister_worker(&worker);
    worker_context_free(ctx);
    return NULL;
}

void worker_threads_stop(void) {
    worker_threads_running = 0;
}
