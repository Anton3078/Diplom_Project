#include "../include/t_master.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_EVENTS 64
#define S_PORT 8080

static spmc_workqueue_t *g_queue = NULL;
void* worker_thread(void *arg);
void worker_threads_stop(void);

static void err_s_exit(const char *msg, int fd) {
    perror(msg);
    if (fd >= 0) close(fd);
    exit(EXIT_FAILURE);
}
static void err_exit(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main() {
    int s_sock_fd, c_sock_fd;
    struct sockaddr_in6 svaddr, claddr;
    socklen_t c_addr_size = sizeof(claddr);
    int epoll_fd, nfds;
    struct epoll_event ev, events[MAX_EVENTS];
    pthread_t threads[MAX_WORKERS];
    unsigned char buff[MAX_IMAGE_SIZE];

    fprintf(stderr, "[Master] Initializing SPMC Queue...\n");
    g_queue = workqueue_create(256, 512);
    if (!g_queue) err_exit("ОШИБКА: workqueue_create");

    s_sock_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (s_sock_fd < 0) err_exit("ОШИБКА: socket");

    int opt = 1;
    if (setsockopt(s_sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
        err_s_exit("ОШИБКА: setsockopt", s_sock_fd);

    memset(&svaddr, 0, sizeof(svaddr));
    svaddr.sin6_family = AF_INET6;
    svaddr.sin6_addr = in6addr_any;
    svaddr.sin6_port = htons(S_PORT);

    if (bind(s_sock_fd, (struct sockaddr*)&svaddr, sizeof(svaddr)) < 0)
        err_s_exit("ОШИБКА: bind", s_sock_fd);
    if (listen(s_sock_fd, SOMAXCONN) < 0)
        err_s_exit("ОШИБКА: listen", s_sock_fd);

    epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) err_s_exit("ОШИБКА: epoll_create1", s_sock_fd);

    ev.events = EPOLLIN;
    ev.data.fd = s_sock_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, s_sock_fd, &ev) < 0)
        err_s_exit("ОШИБКА: epoll_ctl", s_sock_fd);

    const char* detection_model = "./models/detector_v20.4_final.onnx";
    const char* recognition_model = "./models/inception_embedding.onnx";

    fprintf(stderr, "[Master] Spawning %d worker threads...\n", MAX_WORKERS);
    worker_config_t worker_cfgs[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        worker_cfgs[i].queue = g_queue;
        worker_cfgs[i].detection_model_path = detection_model;
        worker_cfgs[i].recognition_model_path = recognition_model;
        if (pthread_create(&threads[i], NULL, worker_thread, &worker_cfgs[i]) != 0)
            err_s_exit("ОШИБКА: pthread_create", s_sock_fd);
    }

    printf("=== SERVER START ===\n");
    fflush(stdout);

    while (1) {
        nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            err_s_exit("ОШИБКА: epoll_wait", s_sock_fd);
        }
        for (int n = 0; n < nfds; n++) {
            if (events[n].data.fd == s_sock_fd) {
                c_sock_fd = accept(s_sock_fd, (struct sockaddr*)&claddr, &c_addr_size);
                if (c_sock_fd < 0) { perror("ОШИБКА: accept"); continue; }

                uint32_t img_len_net = 0;
                ssize_t len_recv = recv(c_sock_fd, &img_len_net, 4, MSG_WAITALL);
                if (len_recv != 4) {
                    fprintf(stderr, "[Master] FD %d: Failed to read image length. Closing.\n", c_sock_fd);
                    close(c_sock_fd); continue;
                }
                uint32_t img_len = ntohl(img_len_net);
                fprintf(stderr, "[Master] FD %d: Client declared image size: %u bytes\n", c_sock_fd, img_len);

                if (img_len == 0 || img_len > MAX_IMAGE_SIZE) {
                    fprintf(stderr, "[Master] FD %d: Invalid image size. Closing.\n", c_sock_fd);
                    close(c_sock_fd); continue;
                }

                ssize_t total_recv = 0;
                while (total_recv < img_len) {
                    ssize_t r = recv(c_sock_fd, buff + total_recv, img_len - total_recv, 0);
                    if (r <= 0) { perror("ОШИБКА: recv data"); break; }
                    total_recv += r;
                }
                fprintf(stderr, "[Master] FD %d: Successfully received %zd/%u bytes\n", c_sock_fd, total_recv, img_len);
                if (total_recv < img_len) { close(c_sock_fd); continue; }

                client_info_t client;
                client.socket_fd = c_sock_fd;
                memcpy(&client.addr, &claddr, sizeof(claddr));
                client.addr_len = c_addr_size;
                client.client_id = 0;

                fprintf(stderr, "[Master] FD %d: Enqueuing task...\n", c_sock_fd);
                if (!workqueue_enqueue(g_queue, buff, (uint32_t)total_recv, &client)) {
                    const char *err_msg = "Server busy";
                    send(c_sock_fd, err_msg, strlen(err_msg), MSG_NOSIGNAL);
                    close(c_sock_fd);
                    fprintf(stderr, "[Master] FD %d: Dropped task (queue full)\n", c_sock_fd);
                }
            }
        }
    }

    worker_threads_stop();
    for (int i = 0; i < MAX_WORKERS; i++) pthread_join(threads[i], NULL);
    close(s_sock_fd);
    workqueue_destroy(g_queue);
    return 0;
}
