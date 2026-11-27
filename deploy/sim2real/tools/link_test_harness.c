/* 协议交叉验证用的 C 侧测试宿主（不参与固件编译，只在 PC 上用 gcc 编译）
 *
 * 用途：把固件里的 lgdog_link.c【原封不动】编译出来，由 Python 驱动做逐字节比对。
 * 这样能保证 C 端与 Python 端的帧编解码永远一致 —— 光看代码"长得一样"不算验证。
 *
 * stdin 指令（每行一条）：
 *   E <sof> <msg_id> <payload_hex>    打包，输出  ENC <frame_hex>
 *   P <sof> <stream_hex>              解帧（每次重建 parser），输出 PARSE 行
 *   Z                                 打印各载荷结构体的 sizeof
 *   S <sof>                           新建一个【保持状态】的 parser 会话（用于测半包）
 *   PN <sof> <stream_hex>             用当前会话 parser 继续解帧（不重建）
 *   Q                                 退出
 *
 *   PARSE 行格式：PARSE <n> [<msg_id>:<payload_hex> ...] CRC=<n> LEN=<n> SOF=<n>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lgdog_link.h"

static int hex2bytes(const char *hex, uint8_t *out, int cap)
{
    int n = 0;
    while (hex[0] && hex[1] && n < cap) {
        unsigned v;
        if (sscanf(hex, "%2x", &v) != 1) break;
        out[n++] = (uint8_t)v;
        hex += 2;
    }
    return n;
}

static void print_hex(const uint8_t *b, int n)
{
    int i;
    for (i = 0; i < n; i++) printf("%02x", b[i]);
}

int main(void)
{
    char line[8192];
    static uint8_t payload[LGD_LINK_MAX_PAYLOAD];
    static uint8_t frame[LGD_LINK_MAX_PAYLOAD + 8];
    static uint8_t stream[8192];
    static LgdLinkMessage msgs[64];
    LgdLinkParser parser;

    /* 关键：stdout 重定向到管道时默认是【块缓冲】，Python 侧 readline 会一直等。
       必须改成行缓冲（或无缓冲），否则这个测试必然挂住。 */
    setvbuf(stdout, NULL, _IOLBF, 0);

    while (fgets(line, sizeof(line), stdin)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (line[0] == 'E') {
            unsigned sof, mid;
            char *hex = strchr(line, ' ');          /* sof */
            hex = strchr(hex + 1, ' ');             /* mid */
            hex = strchr(hex + 1, ' ');             /* payload */
            int plen = (hex && hex[1]) ? hex2bytes(hex + 1, payload, sizeof(payload)) : 0;
            sscanf(line, "E %x %x", &sof, &mid);
            uint16_t n = LgdLink_Encode(frame, sizeof(frame), (uint8_t)sof,
                                        (uint8_t)mid, payload, (uint8_t)plen);
            printf("ENC ");
            print_hex(frame, n);
            printf("\n");
        } else if (line[0] == 'P' && line[1] == 'N') {
            unsigned sof;
            char *sp1 = strchr(line, ' ');
            char *hex = sp1 ? strchr(sp1 + 1, ' ') : NULL;
            sscanf(line, "PN %x", &sof);
            int n = (hex && hex[1]) ? hex2bytes(hex + 1, stream, sizeof(stream)) : 0;
            uint16_t got = LgdLink_Parse(&parser, stream, (uint16_t)n, msgs, 64);
            printf("PARSE %u", got);
            for (uint16_t i = 0; i < got; i++) {
                printf(" %02x:", msgs[i].msg_id);
                print_hex(msgs[i].payload, msgs[i].len);
            }
            printf(" CRC=%u LEN=%u SOF=%u\n",
                   parser.crc_errors, parser.len_errors, parser.sof_drops);
        } else if (line[0] == 'P') {
            unsigned sof;
            /* ★ 注意要取【第二个】空格之后的十六进制串：
               "P a5 <hex...>" 的第一个空格在 P 和 sof 之间，取错会把 sof 也当数据。 */
            char *sp1 = strchr(line, ' ');
            char *hex = sp1 ? strchr(sp1 + 1, ' ') : NULL;
            sscanf(line, "P %x", &sof);
            int n = (hex && hex[1]) ? hex2bytes(hex + 1, stream, sizeof(stream)) : 0;
            LgdLink_ParserInit(&parser, (uint8_t)sof);
            uint16_t got = LgdLink_Parse(&parser, stream, (uint16_t)n, msgs, 64);
            printf("PARSE %u", got);
            for (uint16_t i = 0; i < got; i++) {
                printf(" %02x:", msgs[i].msg_id);
                print_hex(msgs[i].payload, msgs[i].len);
            }
            printf(" CRC=%u LEN=%u SOF=%u\n",
                   parser.crc_errors, parser.len_errors, parser.sof_drops);
        } else if (line[0] == 'S') {
            unsigned sof;
            sscanf(line, "S %x", &sof);
            LgdLink_ParserInit(&parser, (uint8_t)sof);
            printf("OK\n");   /* ★ 必须回一行：调用方按"一行输入一行输出"读取 */
        } else if (line[0] == 'Z') {
            /* 打印各载荷结构体的 sizeof —— 让 Python 侧核对"线上长度"。
               ★ 这一条是为了锁住"结构体对齐填充把长度撑大"的坑：
                 未加 packed 时 ImuPayload 会变成 60、MotorChunk 变成 108，
                 上位机会按长度判为坏帧，表现成"什么数据都收不到"。 */
            printf("SIZES %u %u %u %u %u\n",
                   (unsigned)sizeof(LgdMitCmdPayload),
                   (unsigned)sizeof(LgdBrakePayload),
                   (unsigned)sizeof(LgdImuPayload),
                   (unsigned)sizeof(LgdMotorChunkPayload),
                   (unsigned)sizeof(LgdStatusPayload));
        } else if (line[0] == 'Q') {
            break;
        }
    }
    return 0;
}
