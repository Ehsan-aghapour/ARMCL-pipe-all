#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include<sys/ioctl.h>
#include<iostream>
#include <fstream>
#include <inttypes.h>


#define Pandoon_MAGIC (0xAA)
#define next_state _IO(Pandoon_MAGIC,'c')
#define capture_freqs _IOR(Pandoon_MAGIC,'n',struct f )
#define Apply_freqs _IOW(Pandoon_MAGIC,'a', union ptr)

/*Ehsan: data frequency strucutre for kernel IOCTL API *************************/
struct f{
        uint64_t gf;
        uint32_t f1;
        uint32_t f2;
	uint64_t  capturing=0;
};


//Ehsan pointer wrapper for IOCTL
union ptr{
	int8_t* a;
	uint64_t padding;
};


uint64_t diff_time(timespec start, timespec stop);
int open_pandoon();




int get_freq(int fd);


int set_freq(int fd, int Little, int big, int GPU);
int set_freq(int fd, std::string s);

int init_hikey();
int init_rockpi();
void read_freqs();
int m();
