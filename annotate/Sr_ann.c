/**
 * Copyright (c) 2014-2016, Arm Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/**
 * Copyright (c) 2014-2016, Arm Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef STREAMLINE_ANNOTATE_H
#define STREAMLINE_ANNOTATE_H

/*
 *  User-space only macros:
 *  ANNOTATE_DEFINE            Deprecated and no longer used
 *  ANNOTATE_SETUP             Execute at the start of the program before other ANNOTATE macros are called
 *  ANNOTATE_DELTA_COUNTER     Define a delta counter for use
 *  ANNOTATE_ABSOLUTE_COUNTER  Define an absolute counter for use
 *  ANNOTATE_COUNTER_VALUE     Emit a counter value
 *  CAM_TRACK                  Create a new custom activity map track
 *  CAM_JOB                    Add a new job to a CAM track, use gator_get_time() to obtain the time in nanoseconds
 *  CAM_VIEW_NAME              Name the custom activity map view
 *
 *  User-space and Kernel-space macros:
 *  ANNOTATE(str)                                String annotation
 *  ANNOTATE_CHANNEL(channel, str)               String annotation on a channel
 *  ANNOTATE_COLOR(color, str)                   String annotation with color
 *  ANNOTATE_CHANNEL_COLOR(channel, color, str)  String annotation on a channel with color
 *  ANNOTATE_END()                               Terminate an annotation
 *  ANNOTATE_CHANNEL_END(channel)                Terminate an annotation on a channel
 *  ANNOTATE_NAME_CHANNEL(channel, group, str)   Name a channel and link it to a group
 *  ANNOTATE_NAME_GROUP(group, str)              Name a group
 *  ANNOTATE_VISUAL(data, length, str)           Image annotation with optional string
 *  ANNOTATE_MARKER()                            Marker annotation
 *  ANNOTATE_MARKER_STR(str)                     Marker annotation with a string
 *  ANNOTATE_MARKER_COLOR(color)                 Marker annotation with a color
 *  ANNOTATE_MARKER_COLOR_STR(color, str)        Marker annotation with a string and color
 *
 *  Channels and groups are defined per thread. This means that if the same
 *  channel number is used on different threads they are in fact separate
 *  channels. A channel can belong to only one group per thread. This means
 *  channel 1 cannot be part of both group 1 and group 2 on the same thread.
 *
 *  NOTE: Kernel annotations are not supported in interrupt context.
 */

/* ESC character, hex RGB (little endian) */
#define ANNOTATE_RED    0x0000ff1b
#define ANNOTATE_BLUE   0xff00001b
#define ANNOTATE_GREEN  0x00ff001b
#define ANNOTATE_PURPLE 0xff00ff1b
#define ANNOTATE_YELLOW 0x00ffff1b
#define ANNOTATE_CYAN   0xffff001b
#define ANNOTATE_WHITE  0xffffff1b
#define ANNOTATE_LTGRAY 0xbbbbbb1b
#define ANNOTATE_DKGRAY 0x5555551b
#define ANNOTATE_BLACK  0x0000001b

#define ANNOTATE_COLOR_CYCLE 0x00000000
#define ANNOTATE_COLOR_T1    0x00000001
#define ANNOTATE_COLOR_T2    0x00000002
#define ANNOTATE_COLOR_T3    0x00000003
#define ANNOTATE_COLOR_T4    0x00000004

#ifdef __KERNEL__  /* Start of kernel-space macro definitions */

#include <linux/module.h>

void gator_annotate(const char* str);
void gator_annotate_channel(int channel, const char* str);
void gator_annotate_color(int color, const char* str);
void gator_annotate_channel_color(int channel, int color, const char* str);
void gator_annotate_end(void);
void gator_annotate_channel_end(int channel);
void gator_annotate_name_channel(int channel, int group, const char* str);
void gator_annotate_name_group(int group, const char* str);
void gator_annotate_visual(const char* data, unsigned int length, const char* str);
void gator_annotate_marker(void);
void gator_annotate_marker_str(const char* str);
void gator_annotate_marker_color(int color);
void gator_annotate_marker_color_str(int color, const char* str);

#define ANNOTATE_INVOKE(func, args) \
    func##_ptr = symbol_get(gator_##func); \
    if (func##_ptr) { \
        func##_ptr args; \
        symbol_put(gator_##func); \
    } \

#define ANNOTATE(str) do { \
    void (*annotate_ptr)(const char*); \
    ANNOTATE_INVOKE(annotate, (str)); \
    } while(0)

#define ANNOTATE_CHANNEL(channel, str) do { \
    void (*annotate_channel_ptr)(int, const char*); \
    ANNOTATE_INVOKE(annotate_channel, (channel, str)); \
    } while(0)

#define ANNOTATE_COLOR(color, str) do { \
    void (*annotate_color_ptr)(int, const char*); \
    ANNOTATE_INVOKE(annotate_color, (color, str)); \
    } while(0)

#define ANNOTATE_CHANNEL_COLOR(channel, color, str) do { \
    void (*annotate_channel_color_ptr)(int, int, const char*); \
    ANNOTATE_INVOKE(annotate_channel_color, (channel, color, str)); \
    } while(0)

#define ANNOTATE_END() do { \
    void (*annotate_end_ptr)(void); \
    ANNOTATE_INVOKE(annotate_end, ()); \
    } while(0)

#define ANNOTATE_CHANNEL_END(channel) do { \
    void (*annotate_channel_end_ptr)(int); \
    ANNOTATE_INVOKE(annotate_channel_end, (channel)); \
    } while(0)

#define ANNOTATE_NAME_CHANNEL(channel, group, str) do { \
    void (*annotate_name_channel_ptr)(int, int, const char*); \
    ANNOTATE_INVOKE(annotate_name_channel, (channel, group, str)); \
    } while(0)

#define ANNOTATE_NAME_GROUP(group, str) do { \
    void (*annotate_name_group_ptr)(int, const char*); \
    ANNOTATE_INVOKE(annotate_name_group, (group, str)); \
    } while(0)

#define ANNOTATE_VISUAL(data, length, str) do { \
    void (*annotate_visual_ptr)(const char*, unsigned int, const char*); \
    ANNOTATE_INVOKE(annotate_visual, (data, length, str)); \
    } while(0)

#define ANNOTATE_MARKER() do { \
    void (*annotate_marker_ptr)(void); \
    ANNOTATE_INVOKE(annotate_marker, ()); \
    } while(0)

#define ANNOTATE_MARKER_STR(str) do { \
    void (*annotate_marker_str_ptr)(const char*); \
    ANNOTATE_INVOKE(annotate_marker_str, (str)); \
    } while(0)

#define ANNOTATE_MARKER_COLOR(color) do { \
    void (*annotate_marker_color_ptr)(int); \
    ANNOTATE_INVOKE(annotate_marker_color, (color)); \
    } while(0)

#define ANNOTATE_MARKER_COLOR_STR(color, str) do { \
    void (*annotate_marker_color_str_ptr)(int, const char*); \
    ANNOTATE_INVOKE(annotate_marker_color_str, (color, str)); \
    } while(0)

#else  /* Start of user-space macro definitions */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum gator_annotate_counter_class {
  ANNOTATE_DELTA = 1,
  ANNOTATE_ABSOLUTE,
  ANNOTATE_ACTIVITY
};

enum gator_annotate_display {
  ANNOTATE_AVERAGE = 1,
  ANNOTATE_ACCUMULATE,
  ANNOTATE_HERTZ,
  ANNOTATE_MAXIMUM,
  ANNOTATE_MINIMUM
};

enum gator_annotate_series_composition {
  ANNOTATE_STACKED = 1,
  ANNOTATE_OVERLAY,
  ANNOTATE_LOG10
};

enum gator_annotate_rendering_type {
  ANNOTATE_FILL = 1,
  ANNOTATE_LINE,
  ANNOTATE_BAR
};

void gator_annotate_setup(void);
uint64_t gator_get_time(void);
void gator_annotate_fork_child(void);
void gator_annotate_flush(void);
void gator_annotate_str(const uint32_t channel, const char *const str);
void gator_annotate_color(const uint32_t channel, const uint32_t color, const char *const str);
void gator_annotate_name_channel(const uint32_t channel, const uint32_t group, const char *const str);
void gator_annotate_name_group(const uint32_t group, const char *const str);
void gator_annotate_visual(const void *const data, const uint32_t length, const char *const str);
void gator_annotate_marker(const char *const str);
void gator_annotate_marker_color(const uint32_t color, const char *const str);
void gator_annotate_counter(const uint32_t id, const char *const title, const char *const name, const int per_cpu, const enum gator_annotate_counter_class counter_class, const enum gator_annotate_display display, const char *const units, const uint32_t modifier, const enum gator_annotate_series_composition series_composition, const enum gator_annotate_rendering_type rendering_type, const int average_selection, const int average_cores, const int percentage, const size_t activity_count, const char *const *const activities, const uint32_t *const activity_colors, const uint32_t cores, const uint32_t color, const char *const description);
void gator_annotate_counter_value(const uint32_t core, const uint32_t id, const uint32_t value);
void gator_annotate_activity_switch(const uint32_t core, const uint32_t id, const uint32_t activity, const uint32_t tid);
void gator_cam_track(const uint32_t view_uid, const uint32_t track_uid, const uint32_t parent_track, const char *const name);
void gator_cam_job(const uint32_t view_uid, const uint32_t job_uid, const char *const name, const uint32_t track, const uint64_t start_time, const uint64_t duration, const uint32_t color, const uint32_t primary_dependency, const size_t dependency_count, const uint32_t *const dependencies);
void gator_cam_job_start(const uint32_t view_uid, const uint32_t job_uid, const char *const name, const uint32_t track, const uint64_t time, const uint32_t color);
void gator_cam_job_set_dependencies(const uint32_t view_uid, const uint32_t job_uid, const uint64_t time, const uint32_t primary_dependency, const size_t dependency_count, const uint32_t *const dependencies);
void gator_cam_job_stop(const uint32_t view_uid, const uint32_t job_uid, const uint64_t time);
void gator_cam_view_name(const uint32_t view_uid, const char *const name);

#define ANNOTATE_DEFINE extern int gator_annotate_unused

#define ANNOTATE_SETUP gator_annotate_setup()

#define ANNOTATE(str) gator_annotate_str(0, str)
#define ANNOTATE_CHANNEL(channel, str) gator_annotate_str(channel, str)
#define ANNOTATE_COLOR(color, str) gator_annotate_color(0, color, str)
#define ANNOTATE_CHANNEL_COLOR(channel, color, str) gator_annotate_color(channel, color, str)
#define ANNOTATE_END() gator_annotate_str(0, NULL)
#define ANNOTATE_CHANNEL_END(channel) gator_annotate_str(channel, NULL)
#define ANNOTATE_NAME_CHANNEL(channel, group, str) gator_annotate_name_channel(channel, group, str)
#define ANNOTATE_NAME_GROUP(group, str) gator_annotate_name_group(group, str)
#define ANNOTATE_VISUAL(data, length, str) gator_annotate_visual(data, length, str)
#define ANNOTATE_MARKER() gator_annotate_marker(NULL)
#define ANNOTATE_MARKER_STR(str) gator_annotate_marker(str)
#define ANNOTATE_MARKER_COLOR(color) gator_annotate_marker_color(color, NULL)
#define ANNOTATE_MARKER_COLOR_STR(color, str) gator_annotate_marker_color(color, str)
#define ANNOTATE_DELTA_COUNTER(id, title, name) gator_annotate_counter(id, title, name, 0, ANNOTATE_DELTA, ANNOTATE_ACCUMULATE, NULL, 1, ANNOTATE_STACKED, ANNOTATE_FILL, 0, 0, 0, 0, NULL, NULL, 0, ANNOTATE_COLOR_CYCLE, NULL)
#define ANNOTATE_ABSOLUTE_COUNTER(id, title, name) gator_annotate_counter(id, title, name, 0, ANNOTATE_ABSOLUTE, ANNOTATE_MAXIMUM, NULL, 1, ANNOTATE_STACKED, ANNOTATE_FILL, 0, 0, 0, 0, NULL, NULL, 0, ANNOTATE_COLOR_CYCLE, NULL)
#define ANNOTATE_COUNTER_VALUE(id, value) gator_annotate_counter_value(0, id, value)
#define CAM_TRACK(view_uid, track_uid, parent_track, name) gator_cam_track(view_uid, track_uid, parent_track, name)
#define CAM_JOB(view_uid, job_uid, name, track, start_time, duration, color) gator_cam_job(view_uid, job_uid, name, track, start_time, duration, color, -1, 0, 0)
#define CAM_JOB_DEP(view_uid, job_uid, name, track, start_time, duration, color, dependency) { \
    uint32_t __dependency = dependency; \
    gator_cam_job(view_uid, job_uid, name, track, start_time, duration, color, -1, 1, &__dependency); \
}
#define CAM_JOB_DEPS(view_uid, job_uid, name, track, start_time, duration, color, dependency_count, dependencies) gator_cam_job(view_uid, job_uid, name, track, start_time, duration, color, -1, dependency_count, dependencies)
#define CAM_JOB_START(view_uid, job_uid, name, track, time, color) gator_cam_job_start(view_uid, job_uid, name, track, time, color)
#define CAM_JOB_SET_DEP(view_uid, job_uid, time, dependency) { \
    uint32_t __dependency = dependency; \
    gator_cam_job_set_dependencies(view_uid, job_uid, time, -1, 1, &__dependency); \
}
#define CAM_JOB_SET_DEPS(view_uid, job_uid, time, dependency_count, dependencies) gator_cam_job_set_dependencies(view_uid, job_uid, time, -1, dependency_count, dependencies)
#define CAM_JOB_STOP(view_uid, job_uid, time) gator_cam_job_stop(view_uid, job_uid, time)
#define CAM_VIEW_NAME(view_uid, name) gator_cam_view_name(view_uid, name)

#ifdef __cplusplus
}
#endif

#endif /* _KERNEL_ */
#endif /* STREAMLINE_ANNOTATE_H */

// end .h

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//#include "streamline_annotate.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define THREAD_BUFFER_SIZE (1<<16)
#define THREAD_BUFFER_MASK (THREAD_BUFFER_SIZE - 1)

#define STREAMLINE_ANNOTATE_PARENT "\0streamline-annotate-parent"
#define STREAMLINE_ANNOTATE "\0streamline-annotate"

static const char gator_annotate_handshake[] = "ANNOTATE 4\n";
static const int gator_minimum_version = 24;

static const uint8_t HEADER_UTF8              = 0x01;
static const uint8_t HEADER_UTF8_COLOR        = 0x02;
static const uint8_t HEADER_CHANNEL_NAME      = 0x03;
static const uint8_t HEADER_GROUP_NAME        = 0x04;
static const uint8_t HEADER_VISUAL            = 0x05;
static const uint8_t HEADER_MARKER            = 0x06;
static const uint8_t HEADER_MARKER_COLOR      = 0x07;
static const uint8_t HEADER_COUNTER           = 0x08;
static const uint8_t HEADER_COUNTER_VALUE     = 0x09;
static const uint8_t HEADER_ACTIVITY_SWITCH   = 0x0a;
static const uint8_t HEADER_CAM_TRACK         = 0x0b;
static const uint8_t HEADER_CAM_JOB           = 0x0c;
static const uint8_t HEADER_CAM_VIEW_NAME     = 0x0d;
static const uint8_t HEADER_CAM_JOB_START     = 0x0e;
static const uint8_t HEADER_CAM_JOB_SET_DEPS  = 0x0f;
static const uint8_t HEADER_CAM_JOB_STOP      = 0x10;

static const uint32_t SIZE_COLOR        =  4;
static const uint32_t MAXSIZE_PACK_INT  =  5;
static const uint32_t MAXSIZE_PACK_LONG = 10;

static const uint64_t NS_PER_S = 1000000000;

struct gator_thread {
    struct gator_thread *next;
    const char *oob_data;
    /* oob_data must be written before oob_length */
    size_t oob_length;
    /* Posted when data is sent */
    sem_t sem;
    int fd;
    int tid;
    uint32_t write_pos;
    uint32_t read_pos;
    bool exited;
    char buf[THREAD_BUFFER_SIZE];
};

struct gator_counter {
    struct gator_counter *next;
    const char *title;
    const char *name;
    const char *units;
    const char *description;
    const char **activities;
    uint32_t *activity_colors;
    size_t activity_count;
    int per_cpu;
    int average_selection;
    int average_cores;
    int percentage;
    enum gator_annotate_counter_class counter_class;
    enum gator_annotate_display display;
    enum gator_annotate_series_composition series_composition;
    enum gator_annotate_rendering_type rendering_type;
    uint32_t id;
    uint32_t modifier;
    uint32_t cores;
    uint32_t color;
};

struct gator_cam_track {
    struct gator_cam_track *next;
    const char *name;
    uint32_t view_uid;
    uint32_t track_uid;
    uint32_t parent_track;
};

struct gator_cam_name {
    struct gator_cam_name *next;
    const char *name;
    uint32_t view_uid;
};

struct gator_state {
    struct gator_thread *threads;
    struct gator_counter *counters;
    struct gator_cam_track *cam_tracks;
    struct gator_cam_name *cam_names;
    /* Post to request asynchronous send of data */
    sem_t sender_sem;
    /* Post to request synchronous send of data */
    sem_t sync_sem;
    /* Posted when synchronous send of data has completed */
    sem_t sync_waiter_sem;
    pthread_key_t key;
    int parent_fd;
    bool initialized;
    bool capturing;
    bool forked;
    bool resend_state;
};

static struct gator_state gator_state;
/* Intentionally exported */
uint8_t gator_dont_mangle_keys;

static int gator_socket_cloexec(int domain, int type, int protocol) {
#ifdef SOCK_CLOEXEC
    return socket(domain, type | SOCK_CLOEXEC, protocol);
#else
    const int sock = socket(domain, type, protocol);
    if (sock < 0)
        return -1;

    const int fdf = fcntl(sock, F_GETFD);
    if ((fdf == -1) || (fcntl(sock, F_SETFD, fdf | FD_CLOEXEC) != 0)) {
        close(sock);
        return -1;
    }
    return sock;
#endif
}

#define gator_log(msg) \
    __gator_log(__func__, __FILE__, __LINE__, msg)

static void __gator_log(const char *const func, const char *const file, const int line, const char *const msg)
{
    fprintf(stderr, "%s(%s:%i): %s\n", func, file, line, msg);
}

#define gator_log_errnum(msg, errnum) \
    __gator_log_errnum(__func__, __FILE__, __LINE__, msg, errnum)

static void __gator_log_errnum(const char *const func, const char *const file, const int line, const char *const msg,
                   const int errnum)
{
    fprintf(stderr, "%s(%s:%i): %s: %s\n", func, file, line, msg, strerror(errnum));
}

static void gator_destructor(void *value)
{
    struct gator_thread *const thread = (struct gator_thread *)value;
    if (thread != NULL)
        thread->exited = true;
}

void gator_annotate_fork_child(void)
{
    /* Single threaded at this point */
    struct gator_thread *thread;
    pthread_setspecific(gator_state.key, NULL);
    for (thread = gator_state.threads; thread != NULL; thread = thread->next) {
        thread->exited = true;
        thread->read_pos = thread->write_pos;
    }

    gator_state.forked = true;
}

void gator_annotate_flush(void)
{
    if (gator_state.capturing) {
        /* Request synchronous send of data */
        sem_post(&gator_state.sync_sem);
        /* Wake up sender */
        sem_post(&gator_state.sender_sem);
        /* Wait for completion */
        sem_wait(&gator_state.sync_waiter_sem);
    }
}

static void gator_set_ts(struct timespec *const ts, const uint64_t time)
{
    ts->tv_sec = time / NS_PER_S;
    ts->tv_nsec = time % NS_PER_S;
}

static uint64_t gator_time(const clockid_t clk_id)
{
    struct timespec ts;
    if (clock_gettime(clk_id, &ts) != 0)
        return ~0;

    return NS_PER_S*ts.tv_sec + ts.tv_nsec;
}

uint64_t gator_get_time(void)
{
#ifndef CLOCK_MONOTONIC_RAW
    /* Android doesn't have this defined but it was added in Linux 2.6.28 */
#define CLOCK_MONOTONIC_RAW 4
#endif
    return gator_time(CLOCK_MONOTONIC_RAW);
}

static bool gator_parent_connect(void)
{
    const int fd = gator_socket_cloexec(PF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return false;

    struct sockaddr_un addr = { 0 };
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, STREAMLINE_ANNOTATE_PARENT, sizeof(STREAMLINE_ANNOTATE_PARENT));
    if (connect(fd, (const struct sockaddr *)&addr,
            offsetof(struct sockaddr_un, sun_path) + sizeof(STREAMLINE_ANNOTATE_PARENT) - 1) != 0) {
        close(fd);
        return false;
    }

    gator_state.parent_fd = fd;
    return true;
}

static uint32_t gator_buf_pos(const uint32_t pos) {
    return pos & THREAD_BUFFER_MASK;
}

static uint32_t gator_buf_write_byte(char *const buf, uint32_t *const write_pos_ptr, char b)
{
    const uint32_t write_pos = *write_pos_ptr;
    buf[write_pos] = b;
    *write_pos_ptr = gator_buf_pos(write_pos + 1);
    return 1;
}

static uint32_t gator_buf_write_uint32(char *const buf, uint32_t *const write_pos_ptr, uint32_t i)
{
    const uint32_t write_pos = *write_pos_ptr;
    buf[gator_buf_pos(write_pos + 0)] = i & 0xff;
    buf[gator_buf_pos(write_pos + 1)] = (i >> 8) & 0xff;
    buf[gator_buf_pos(write_pos + 2)] = (i >> 16) & 0xff;
    buf[gator_buf_pos(write_pos + 3)] = (i >> 24) & 0xff;
    *write_pos_ptr = gator_buf_pos(write_pos + sizeof(uint32_t));
    return sizeof(uint32_t);
}

static uint32_t gator_buf_write_color(char *const buf, uint32_t *const write_pos_ptr, const uint32_t color)
{
    const uint32_t write_pos = *write_pos_ptr;
    buf[gator_buf_pos(write_pos + 0)] = (color >> 8) & 0xff;
    buf[gator_buf_pos(write_pos + 1)] = (color >> 16) & 0xff;
    buf[gator_buf_pos(write_pos + 2)] = (color >> 24) & 0xff;
    buf[gator_buf_pos(write_pos + 3)] = (color >> 0) & 0xff;
    *write_pos_ptr = gator_buf_pos(write_pos + 4);
    return 4;
}

static uint32_t gator_buf_write_int(char *const buf, uint32_t *const write_pos_ptr, int32_t x)
{
    const uint32_t write_pos = *write_pos_ptr;
    int packed_bytes = 0;
    int more = true;

    while (more) {
        /* low order 7 bits of x */
        char b = x & 0x7f;

        x >>= 7;

        if ((x == 0 && (b & 0x40) == 0) || (x == -1 && (b & 0x40) != 0))
            more = false;
        else
            b |= 0x80;

        buf[gator_buf_pos(write_pos + packed_bytes)] = b;
        packed_bytes++;
    }

    *write_pos_ptr = gator_buf_pos(write_pos + packed_bytes);
    return packed_bytes;
}

static uint32_t gator_buf_write_long(char *const buf, uint32_t *const write_pos_ptr, int64_t x)
{
    const uint32_t write_pos = *write_pos_ptr;
    int packed_bytes = 0;
    int more = true;

    while (more) {
        /* low order 7 bits of x */
        char b = x & 0x7f;

        x >>= 7;

        if ((x == 0 && (b & 0x40) == 0) || (x == -1 && (b & 0x40) != 0))
            more = false;
        else
            b |= 0x80;

        buf[gator_buf_pos(write_pos + packed_bytes)] = b;
        packed_bytes++;
    }

    *write_pos_ptr = gator_buf_pos(write_pos + packed_bytes);
    return packed_bytes;
}

static uint32_t gator_buf_write_time(char *const buf, uint32_t *const write_pos_ptr)
{
    return gator_buf_write_long(buf, write_pos_ptr, gator_get_time());
}

static uint32_t gator_buf_write_bytes(char *const buf, uint32_t *const write_pos_ptr, const char *const data,
                      const uint32_t count)
{
    const uint32_t write_pos = *write_pos_ptr;
    if (write_pos + count <= THREAD_BUFFER_SIZE) {
        memcpy(buf + write_pos, data, count);
    } else {
        const uint32_t first = THREAD_BUFFER_SIZE - write_pos;
        const uint32_t second = count - first;
        memcpy(buf + write_pos, data, first);
        memcpy(buf, data + first, second);
    }
    *write_pos_ptr = gator_buf_pos(write_pos + count);
    return count;
}

static int gator_connect(const int tid)
{
    const int fd = gator_socket_cloexec(PF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;

    struct sockaddr_un addr = { 0 };
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, STREAMLINE_ANNOTATE, sizeof(STREAMLINE_ANNOTATE));
    if (connect(fd, (const struct sockaddr *)&addr,
            offsetof(struct sockaddr_un, sun_path) + sizeof(STREAMLINE_ANNOTATE) - 1) != 0) {
        close(fd);
        return -1;
    }

    /* Send tid as gatord cannot autodiscover it and the per process unique id */
    uint32_t write_pos = 0;
    char buf[sizeof(gator_annotate_handshake) + 2*sizeof(uint32_t) + 1];
    gator_buf_write_bytes(buf, &write_pos, gator_annotate_handshake, sizeof(gator_annotate_handshake) - 1);
    gator_buf_write_uint32(buf, &write_pos, tid);
    gator_buf_write_uint32(buf, &write_pos, getpid());
    gator_buf_write_byte(buf, &write_pos, gator_dont_mangle_keys);
    const ssize_t bytes = send(fd, buf, write_pos, MSG_NOSIGNAL);
    if (bytes != (ssize_t)write_pos) {
        close(fd);
        return -1;
    }

    return fd;
}

static void gator_start_capturing(void)
{
    if (__sync_bool_compare_and_swap(&gator_state.capturing, false, true))
        gator_state.resend_state = true;
}

static void gator_stop_capturing(void)
{
    if (__sync_bool_compare_and_swap(&gator_state.capturing, true, false)) {
        struct gator_thread *thread;
        for (thread = gator_state.threads; thread != NULL; thread = thread->next) {
            thread->read_pos = thread->write_pos;
            thread->oob_length = 0;
            sem_post(&thread->sem);
            if (thread->fd > 0) {
                close(thread->fd);
                thread->fd = -1;
            }
        }
    }
}

static bool gator_send(struct gator_thread *const thread, const uint32_t write_pos)
{
    size_t write;
    ssize_t bytes;

    if (write_pos > thread->read_pos) {
        write = write_pos - thread->read_pos;
        bytes = send(thread->fd, thread->buf + thread->read_pos, write, MSG_NOSIGNAL);
        if (bytes <= 0)
            return false;
        thread->read_pos = gator_buf_pos(thread->read_pos + bytes);
    } else {
        write = THREAD_BUFFER_SIZE - thread->read_pos;
        bytes = send(thread->fd, thread->buf + thread->read_pos, write, MSG_NOSIGNAL);
        if (bytes <= 0)
            return false;
        thread->read_pos = gator_buf_pos(thread->read_pos + bytes);

        if (write == (size_t)bytes) {
            /* Don't write more on a short write to be fair to other threads */
            write = write_pos;
            bytes = send(thread->fd, thread->buf, write, MSG_NOSIGNAL);
            if (bytes <= 0)
                return false;
            thread->read_pos = gator_buf_pos(thread->read_pos + bytes);
        }
    }

    if (write == (size_t)bytes && thread->oob_length > 0) {
        __sync_synchronize();
        /* Don't write more on a short write to be fair to other threads */
        bytes = send(thread->fd, thread->oob_data, thread->oob_length, MSG_NOSIGNAL);
        if (bytes <= 0)
            return false;
        thread->oob_data += bytes;
        thread->oob_length -= bytes;
    }

    return true;
}

static void* gator_func(void *arg)
{
    bool print = true;
    uint64_t last = 0;

    prctl(PR_SET_NAME, (unsigned long)&"gator-annotate", 0, 0, 0);

    if (arg != NULL) {
        /* Forked */
        if (gator_state.parent_fd >= 0) {
            close(gator_state.parent_fd);
            gator_state.parent_fd = -1;
        }
    }

    for (;;) {
        if (gator_state.parent_fd < 0) {
            if (gator_parent_connect()) {
                /* Optimistically begin capturing data */
                gator_start_capturing();
            } else {
                gator_stop_capturing();
                if (print) {
                    fprintf(stderr,
                        "Warning %s(%s:%i): Not connected to gatord, the application will run normally but Streamline will not collect annotations. To collect annotations, please verify you are running gatord 5.%i or later and that SELinux is disabled.\n",
                        __func__, __FILE__, __LINE__, gator_minimum_version);
                    print = false;
                }
                sleep(1);
                continue;
            }
        }

        if (!gator_state.capturing) {
            char temp;
            const ssize_t bytes = read(gator_state.parent_fd, &temp, sizeof(temp));
            if (bytes <= 0) {
                close(gator_state.parent_fd);
                gator_state.parent_fd = -1;
                continue;
            }
            gator_start_capturing();
            gator_state.resend_state = true;
        }

        if (gator_state.capturing) {
            /* Iterate every 100ms */
            const uint64_t freq = NS_PER_S/10;
            const uint64_t now = gator_time(CLOCK_REALTIME);
            const uint64_t wakeup = last + freq;
            if (wakeup < now) {
                /* Already timed out, save the current time and iterate again */
                last = now;
            } else {
                struct timespec ts;
                gator_set_ts(&ts, wakeup);
                sem_timedwait(&gator_state.sender_sem, &ts);
                last = wakeup;
                while (sem_trywait(&gator_state.sender_sem) == 0) {
                    /* Ignore multiple posts */
                }
            }
        }

        int sync_count = 0;
        while (sem_trywait(&gator_state.sync_sem) == 0)
            ++sync_count;

        struct gator_thread **prev = &gator_state.threads;
        struct gator_thread *thread = *prev;
        while (thread != NULL) {
            if (gator_state.capturing) {
                if (thread->fd < 0 && (thread->fd = gator_connect(thread->tid)) < 0) {
                    gator_stop_capturing();
                } else {
                    const uint32_t write_pos = thread->write_pos;
                    if (write_pos != thread->read_pos || thread->oob_length > 0) {
                        if (!gator_send(thread, write_pos))
                            gator_stop_capturing();
                        else
                            sem_post(&thread->sem);
                    }
                }
            }

            struct gator_thread *const next = thread->next;
            if (!thread->exited || !__sync_bool_compare_and_swap(prev, thread, next)) {
                /* If the cas fails, the linked list has changed, get it next time */
                prev = &thread->next;
                thread = *prev;
            } else {
                if (thread->fd > 0)
                    close(thread->fd);
                sem_destroy(&thread->sem);
                free(thread);
                thread = next;
            }
        }

        for (; sync_count > 0; --sync_count)
            sem_post(&gator_state.sync_waiter_sem);
    }

    return NULL;
}

void gator_annotate_setup(void)
{
    /* Support calling gator_annotate_setup more than once, but not at the same time on different cores */
    if (__sync_bool_compare_and_swap(&gator_state.initialized, false, true)) {
        gator_state.parent_fd = -1;
        /* Optimistically begin capturing data */
        gator_state.capturing = true;

        int err = sem_init(&gator_state.sender_sem, 0, 0);
        if (err != 0) {
            gator_log_errnum("sem_init failed", err);
            return;
        }

        err = sem_init(&gator_state.sync_sem, 0, 0);
        if (err != 0) {
            gator_log_errnum("sem_init failed", err);
            return;
        }

        err = sem_init(&gator_state.sync_waiter_sem, 0, 0);
        if (err != 0) {
            gator_log_errnum("sem_init failed", err);
            return;
        }

        err = pthread_key_create(&gator_state.key, gator_destructor);
        if (err != 0) {
            gator_log_errnum("pthread_key_create failed", err);
            return;
        }

#if !defined(__ANDROID_API__) || __ANDROID_API__ >= 21
        err = pthread_atfork(NULL, NULL, gator_annotate_fork_child);
        if (err != 0) {
            gator_log_errnum("pthread_atfork failed", err);
            return;
        }
#endif

        if (atexit(gator_annotate_flush) != 0) {
            gator_log("atexit failed");
            return;
        }

        pthread_t thread;
        err = pthread_create(&thread, NULL, gator_func, NULL);
        if (err != 0) {
            gator_log_errnum("pthread_create failed", err);
            return;
        }
    }
}

static void gator_annotate_write_counter(struct gator_thread *const thread, const struct gator_counter *const counter);
static void gator_annotate_write_cam_track(struct gator_thread *const thread,
                       const struct gator_cam_track *const cam_track);
static void gator_annotate_write_cam_name(struct gator_thread *const thread, const struct gator_cam_name *const cam_name);

static struct gator_thread *gator_get_thread(void)
{
    if (__sync_bool_compare_and_swap(&gator_state.forked, true, false)) {
        pthread_t thread;
        int err = pthread_create(&thread, NULL, gator_func, (void *)1);
        if (err != 0) {
            gator_log_errnum("pthread_create failed", err);
            return NULL;
        }
    }

    if (!gator_state.capturing)
        return NULL;

    struct gator_thread *thread = (struct gator_thread *)pthread_getspecific(gator_state.key);
    int err;
    if (thread != NULL)
        goto success;

    thread = (struct gator_thread *)malloc(sizeof(*thread));
    if (thread == NULL) {
        gator_log_errnum("malloc failed", errno);
        return NULL;
    }

    thread->oob_data = NULL;
    thread->oob_length = 0;
    thread->fd = -1;
    thread->tid = syscall(__NR_gettid);
    thread->write_pos = 0;
    thread->read_pos = 0;
    thread->exited = false;

    err = sem_init(&thread->sem, 0, 0);
    if (err != 0) {
        gator_log_errnum("sem_init failed", err);
        goto fail_free_thread;
    }

    err = pthread_setspecific(gator_state.key, thread);
    if (err != 0) {
        gator_log_errnum("pthread_setspecific failed", err);
        goto fail_sem_destroy;
    }

    do
        thread->next = gator_state.threads;
    while (!__sync_bool_compare_and_swap(&gator_state.threads, thread->next, thread));

success:
    if (__sync_bool_compare_and_swap(&gator_state.resend_state, true, false)) {
        struct gator_counter *counter;
        for (counter = gator_state.counters; counter != NULL; counter = counter->next)
            gator_annotate_write_counter(thread, counter);
        struct gator_cam_track *cam_track;
        for (cam_track = gator_state.cam_tracks; cam_track != NULL; cam_track = cam_track->next)
            gator_annotate_write_cam_track(thread, cam_track);
        struct gator_cam_name *cam_name;
        for (cam_name = gator_state.cam_names; cam_name != NULL; cam_name = cam_name->next)
            gator_annotate_write_cam_name(thread, cam_name);
    }

    return thread;

fail_sem_destroy:
    sem_destroy(&thread->sem);
fail_free_thread:
    free(thread);
    return NULL;
}

static uint32_t gator_buf_free(const struct gator_thread *const thread)
{
    return (thread->read_pos - thread->write_pos - 1) & THREAD_BUFFER_MASK;
}

static uint32_t gator_buf_used(const struct gator_thread *const thread)
{
    return (thread->write_pos - thread->read_pos) & THREAD_BUFFER_MASK;
}

#define gator_buf_wait_bytes(thread, bytes) \
    const uint32_t __bytes = (bytes); \
    if (__bytes > THREAD_BUFFER_SIZE/2) { \
        /* Large annotations won't fit in the buffer */ \
        gator_log("message is too large"); \
        return; \
    } \
    __gator_buf_wait_bytes((thread), __bytes);

static void __gator_buf_wait_bytes(struct gator_thread *const thread, const uint32_t bytes)
{
    while (gator_buf_free(thread) < bytes) {
        sem_post(&gator_state.sender_sem);
        sem_wait(&thread->sem);
    }
}

static void gator_msg_begin(const char marker, struct gator_thread *const thread, uint32_t *const write_pos_ptr,
                uint32_t *const size_pos_ptr, uint32_t *const length_ptr)
{
    *write_pos_ptr = thread->write_pos;
    gator_buf_write_byte(thread->buf, write_pos_ptr, marker);
    *size_pos_ptr = *write_pos_ptr;
    *write_pos_ptr = gator_buf_pos(*write_pos_ptr + sizeof(uint32_t));
    *length_ptr = 0;
}

static void gator_msg_end(struct gator_thread *const thread, const uint32_t write_pos, uint32_t size_pos,
                const uint32_t length)
{
    gator_buf_write_uint32(thread->buf, &size_pos, length);
    thread->write_pos = write_pos;

    /* Wakeup the sender thread if 3/4 full */
    if (gator_buf_used(thread) >= 3*THREAD_BUFFER_SIZE/4)
        sem_post(&gator_state.sender_sem);
}

void gator_annotate_str(const uint32_t channel, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + MAXSIZE_PACK_INT + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_UTF8, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, channel);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_color(const uint32_t channel, const uint32_t color, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + MAXSIZE_PACK_INT + SIZE_COLOR + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_UTF8_COLOR, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, channel);
    length += gator_buf_write_color(thread->buf, &write_pos, color);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_name_channel(const uint32_t channel, const uint32_t group, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + 2*MAXSIZE_PACK_INT + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CHANNEL_NAME, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, channel);
    length += gator_buf_write_int(thread->buf, &write_pos, group);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_name_group(const uint32_t group, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + MAXSIZE_PACK_INT + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_GROUP_NAME, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, group);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_visual(const void *const data, const uint32_t data_length, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    /* Don't include data_length because it doesn't end up in the buffer */
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + str_size + 1);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_VISUAL, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);
    length += gator_buf_write_byte(thread->buf, &write_pos, '\0');
    /* Calculate the length, but don't write the image */
    length += data_length;
    /* Write the length and commit the first part of the message */
    gator_buf_write_uint32(thread->buf, &size_pos, length);
    thread->write_pos = write_pos;

    thread->oob_data = (const char *)data;
    __sync_synchronize();
    thread->oob_length = data_length;
    while (thread->oob_length > 0) {
        sem_post(&gator_state.sender_sem);
        sem_wait(&thread->sem);
    }
}

void gator_annotate_marker(const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_MARKER, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_marker_color(const uint32_t color, const char *const str)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int str_size = (str == NULL) ? 0 : strlen(str);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + SIZE_COLOR + str_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_MARKER_COLOR, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_color(thread->buf, &write_pos, color);
    length += gator_buf_write_bytes(thread->buf, &write_pos, str, str_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

static void gator_annotate_write_counter(struct gator_thread *const thread, const struct gator_counter *const counter)
{
    const int title_size = (counter->title == NULL) ? 0 : strlen(counter->title);
    const int name_size = (counter->name == NULL) ? 0 : strlen(counter->name);
    const int units_size = (counter->units == NULL) ? 0 : strlen(counter->units);
    const int description_size = (counter->description == NULL) ? 0 : strlen(counter->description);
    int activity_size = 0;
    size_t i;
    for (i = 0; i < counter->activity_count; ++i) {
        activity_size += (counter->activities[i] == NULL) ? 0 : strlen(counter->activities[i]);
        activity_size += SIZE_COLOR;
    }
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 12*MAXSIZE_PACK_INT + SIZE_COLOR + activity_size +
                 title_size + 1 + name_size + 1 + units_size + 1 + description_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_COUNTER, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, counter->id);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->per_cpu);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->counter_class);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->display);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->modifier);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->series_composition);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->rendering_type);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->average_selection);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->average_cores);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->percentage);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->activity_count);
    length += gator_buf_write_int(thread->buf, &write_pos, counter->cores);
    length += gator_buf_write_color(thread->buf, &write_pos, counter->color);
    for (i = 0; i < counter->activity_count; ++i) {
        length += gator_buf_write_bytes(thread->buf, &write_pos, counter->activities[i],
                        (counter->activities[i] == NULL) ? 0 : strlen(counter->activities[i]));
        length += gator_buf_write_byte(thread->buf, &write_pos, '\0');
        length += gator_buf_write_color(thread->buf, &write_pos, counter->activity_colors[i]);
    }
    length += gator_buf_write_bytes(thread->buf, &write_pos, counter->title, title_size);
    length += gator_buf_write_byte(thread->buf, &write_pos, '\0');
    length += gator_buf_write_bytes(thread->buf, &write_pos, counter->name, name_size);
    length += gator_buf_write_byte(thread->buf, &write_pos, '\0');
    length += gator_buf_write_bytes(thread->buf, &write_pos, counter->units, units_size);
    length += gator_buf_write_byte(thread->buf, &write_pos, '\0');
    length += gator_buf_write_bytes(thread->buf, &write_pos, counter->description, description_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_counter(const uint32_t id, const char *const title, const char *const name, const int per_cpu,
                const enum gator_annotate_counter_class counter_class,
                const enum gator_annotate_display display, const char *const units, const uint32_t modifier,
                const enum gator_annotate_series_composition series_composition,
                const enum gator_annotate_rendering_type rendering_type, const int average_selection,
                const int average_cores, const int percentage, const size_t activity_count,
                const char *const *const activities, const uint32_t *const activity_colors,
                const uint32_t cores, const uint32_t color, const char *const description)
{
    struct gator_counter *const counter = (struct gator_counter *)malloc(sizeof(*counter));
    if (counter == NULL) {
        gator_log_errnum("malloc failed", errno);
        return;
    }

    /* Save off this counter so it can be resent if needed */
    counter->id = id;
    counter->title = (title == NULL) ? NULL : strdup(title);
    counter->name = (name == NULL) ? NULL : strdup(name);
    counter->per_cpu = per_cpu;
    counter->counter_class = counter_class;
    counter->display = display;
    counter->units = (units == NULL) ? NULL : strdup(units);
    counter->modifier = modifier;
    counter->series_composition = series_composition;
    counter->rendering_type = rendering_type;
    counter->average_selection = average_selection;
    counter->average_cores = average_cores;
    counter->percentage = percentage;
    counter->activity_count = activity_count;
    if (activity_count == 0) {
        counter->activities = NULL;
        counter->activity_colors = NULL;
    } else {
        counter->activities = (const char **)malloc(activity_count*sizeof(activities[0]));
        if (counter->activities == NULL) {
            gator_log_errnum("malloc failed", errno);
            goto free_counter;
        }
        counter->activity_colors = (uint32_t *)malloc(activity_count*sizeof(activity_colors[0]));
        if (counter->activity_colors == NULL) {
            gator_log_errnum("malloc failed", errno);
            goto free_activities;
        }
        size_t i;
        for (i = 0; i < activity_count; ++i) {
            counter->activities[i] = (activities[i] == NULL) ? NULL : strdup(activities[i]);
            counter->activity_colors[i] = activity_colors[i];
        }
    }
    counter->cores = cores;
    counter->color = color;
    counter->description = (description == NULL) ? NULL : strdup(description);

    do
        counter->next = gator_state.counters;
    while (!__sync_bool_compare_and_swap(&gator_state.counters, counter->next, counter));

    {
        struct gator_thread *const thread = gator_get_thread();
        if (thread == NULL)
            return;

        gator_annotate_write_counter(thread, counter);
    }

    return;

free_activities:
    free(counter->activities);
free_counter:
    free(counter);
    return;
}

void gator_annotate_counter_value(const uint32_t core, const uint32_t id, const uint32_t value)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + 3*MAXSIZE_PACK_INT);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_COUNTER_VALUE, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, core);
    length += gator_buf_write_int(thread->buf, &write_pos, id);
    length += gator_buf_write_int(thread->buf, &write_pos, value);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_annotate_activity_switch(const uint32_t core, const uint32_t id, const uint32_t activity, const uint32_t tid)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_LONG + 4*MAXSIZE_PACK_INT);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_ACTIVITY_SWITCH, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_time(thread->buf, &write_pos);
    length += gator_buf_write_int(thread->buf, &write_pos, core);
    length += gator_buf_write_int(thread->buf, &write_pos, id);
    length += gator_buf_write_int(thread->buf, &write_pos, activity);
    length += gator_buf_write_int(thread->buf, &write_pos, tid);

    gator_msg_end(thread, write_pos, size_pos, length);
}

static void gator_annotate_write_cam_track(struct gator_thread *const thread,
                       const struct gator_cam_track *const cam_track)
{
    const int name_size = (cam_track->name == NULL) ? 0 : strlen(cam_track->name);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 3*MAXSIZE_PACK_INT + name_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_TRACK, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, cam_track->view_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, cam_track->track_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, cam_track->parent_track);
    length += gator_buf_write_bytes(thread->buf, &write_pos, cam_track->name, name_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_cam_track(const uint32_t view_uid, const uint32_t track_uid, const uint32_t parent_track,
             const char *const name)
{
    struct gator_cam_track *const cam_track = (struct gator_cam_track *)malloc(sizeof(*cam_track));
    if (cam_track == NULL) {
        gator_log_errnum("malloc failed", errno);
        return;
    }

    /* Save off this track so it can be resent if needed */
    cam_track->view_uid = view_uid;
    cam_track->track_uid = track_uid;
    cam_track->parent_track = parent_track;
    cam_track->name = (name == NULL) ? NULL : strdup(name);

    do
        cam_track->next = gator_state.cam_tracks;
    while (!__sync_bool_compare_and_swap(&gator_state.cam_tracks, cam_track->next, cam_track));

    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_annotate_write_cam_track(thread, cam_track);
}

void gator_cam_job(const uint32_t view_uid, const uint32_t job_uid, const char *const name, const uint32_t track,
           const uint64_t start_time, const uint64_t duration, const uint32_t color,
           const uint32_t primary_dependency, const size_t dependency_count, const uint32_t *const dependencies)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int name_size = (name == NULL) ? 0 : strlen(name);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 5*MAXSIZE_PACK_INT + 2*MAXSIZE_PACK_LONG + SIZE_COLOR +
                 dependency_count*MAXSIZE_PACK_INT + name_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_JOB, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, view_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, job_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, track);
    length += gator_buf_write_long(thread->buf, &write_pos, start_time);
    length += gator_buf_write_long(thread->buf, &write_pos, duration);
    length += gator_buf_write_color(thread->buf, &write_pos, color);
    length += gator_buf_write_int(thread->buf, &write_pos, primary_dependency);
    length += gator_buf_write_int(thread->buf, &write_pos, dependency_count);
    size_t i;
    for (i = 0; i < dependency_count; ++i)
        length += gator_buf_write_int(thread->buf, &write_pos, dependencies[i]);
    length += gator_buf_write_bytes(thread->buf, &write_pos, name, name_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_cam_job_start(const uint32_t view_uid, const uint32_t job_uid, const char *const name, const uint32_t track,
                         const uint64_t time, const uint32_t color)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    const int name_size = (name == NULL) ? 0 : strlen(name);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 3*MAXSIZE_PACK_INT + MAXSIZE_PACK_LONG + SIZE_COLOR + name_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_JOB_START, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, view_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, job_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, track);
    length += gator_buf_write_long(thread->buf, &write_pos, time);
    length += gator_buf_write_color(thread->buf, &write_pos, color);
    length += gator_buf_write_bytes(thread->buf, &write_pos, name, name_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_cam_job_set_dependencies(const uint32_t view_uid, const uint32_t job_uid, const uint64_t time,
                                    const uint32_t primary_dependency, const size_t dependency_count,
                                    const uint32_t *const dependencies)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 4*MAXSIZE_PACK_INT + MAXSIZE_PACK_LONG +
                         dependency_count*MAXSIZE_PACK_INT);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_JOB_SET_DEPS, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, view_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, job_uid);
    length += gator_buf_write_long(thread->buf, &write_pos, time);
    length += gator_buf_write_int(thread->buf, &write_pos, primary_dependency);
    length += gator_buf_write_int(thread->buf, &write_pos, dependency_count);
    size_t i;
    for (i = 0; i < dependency_count; ++i)
        length += gator_buf_write_int(thread->buf, &write_pos, dependencies[i]);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_cam_job_stop(const uint32_t view_uid, const uint32_t job_uid, const uint64_t time)
{
    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + 2*MAXSIZE_PACK_INT + MAXSIZE_PACK_LONG);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_JOB_STOP, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, view_uid);
    length += gator_buf_write_int(thread->buf, &write_pos, job_uid);
    length += gator_buf_write_long(thread->buf, &write_pos, time);

    gator_msg_end(thread, write_pos, size_pos, length);
}

static void gator_annotate_write_cam_name(struct gator_thread *const thread, const struct gator_cam_name *const cam_name)
{
    const int name_size = (cam_name->name == NULL) ? 0 : strlen(cam_name->name);
    gator_buf_wait_bytes(thread, 1 + sizeof(uint32_t) + MAXSIZE_PACK_INT + name_size);

    uint32_t write_pos;
    uint32_t size_pos;
    uint32_t length;
    gator_msg_begin(HEADER_CAM_VIEW_NAME, thread, &write_pos, &size_pos, &length);

    length += gator_buf_write_int(thread->buf, &write_pos, cam_name->view_uid);
    length += gator_buf_write_bytes(thread->buf, &write_pos, cam_name->name, name_size);

    gator_msg_end(thread, write_pos, size_pos, length);
}

void gator_cam_view_name(const uint32_t view_uid, const char *const name)
{
    struct gator_cam_name *const cam_name = (struct gator_cam_name *)malloc(sizeof(*cam_name));
    if (cam_name == NULL) {
        return;
    }

    /* Save off this name so it can be resent if needed */
    cam_name->next = NULL;
    cam_name->view_uid = view_uid;
    cam_name->name = (name == NULL) ? NULL : strdup(name);

    do
        cam_name->next = gator_state.cam_names;
    while (!__sync_bool_compare_and_swap(&gator_state.cam_names, cam_name->next, cam_name));

    struct gator_thread *const thread = gator_get_thread();
    if (thread == NULL)
        return;

    gator_annotate_write_cam_name(thread, cam_name);
}
