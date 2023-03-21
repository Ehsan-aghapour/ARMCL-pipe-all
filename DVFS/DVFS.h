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




static
uint64_t diff_time(timespec start, timespec stop){
        uint64_t out;
        out=((uint64_t)stop.tv_sec*(uint64_t)1.0e9+(uint64_t)stop.tv_nsec)-((uint64_t)start.tv_sec*(uint64_t)1.0e9+(uint64_t)start.tv_nsec);
        return out;
}
static
int open_pandoon(){
        int fddd=-1;
        fddd = open("/dev/pandoon_device", O_RDWR);
        if (fddd<0){
                printf("DRPM_Ehsan: Pandoon Not Opened. ERROR CODE:%d, ERROR MEANING:%s\n",errno,strerror(errno));
                close(fddd);
                return -1;
        }
        return fddd;
}

static
int get_freq(int fd){
        static struct f freqs;
        int jj=ioctl(fd,capture_freqs, &freqs);
        //printf("second capture_freqs:%d\n",jj);
        if(jj<0){
                std::cerr<<"error \n";
        }

        printf("Freqs Captured, FA53:%u, FA73:%u, F_GPU:%" PRIu64", Capturing:%" PRIu64,freqs.f1,freqs.f2,freqs.gf,freqs.capturing);
        printf("\n");
        //printf("freqs:%" PRIu64,freqs.capturing);
        //printf("\n");
        return 0;
}

static
int set_freq(int fd, int Little, int big, int GPU){
        static int l,b,g=0;
        if (Little>-1){
                l=Little;
        }
        if (big>-1){
                b=big;
        }
        if(GPU>-1){
                g=GPU;
        }
        union ptr ptr_freqs;
        timespec t_start,t_end;
        std::cout<<"req freqs, little:"<<Little<<", big:"<<big<<", GPU:"<<GPU<<std::endl;
        std::cout<<"apply freqs, little:"<<l<<", big:"<<b<<", GPU:"<<g<<std::endl;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        g=4-g;
        int8_t dfreqs[3];
        dfreqs[0]=g;
        dfreqs[1]=b;
        dfreqs[2]=l;
        ptr_freqs.padding=0;
        ptr_freqs.a=dfreqs;
        int jj=ioctl(fd, Apply_freqs,ptr_freqs.a);
        if(jj<0){
                printf("DRPM_Ehsan: Apply freqs ioctl error, ret=%d, error:%d, meaning: %s\n",jj,errno,strerror(errno));      
                printf("\n");        
                return -1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        unsigned long int ioctltime=ioctltime=diff_time(t_start,t_end);
        return 0;
}
static
int set_freq(int fd, std::string s){
        int Little,big,GPU;
        int fff=std::stoi(s);
        /*GPU=5-int(fff%5);
        big=int(fff/5)%8;
        Little=int(fff/40);*/
        GPU=int(fff%10);
        big=int(fff/10)%10;
        Little=int(fff/100);
        return set_freq(fd, Little, big, GPU);

}
static
int init_hikey(){
        std::string CPU_path="/sys/devices/system/cpu/cpufreq/";
        std::string GPU_path="/sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali";
        std::string IOCTL_path="/dev/pandoon_device";
        std::string Command="chmod 666 " + IOCTL_path;
        Command="echo pandoon > " + CPU_path + "policy4/scaling_governor";
        system(Command.c_str());
        Command="echo pandoon > " + CPU_path + "policy0/scaling_governor";
        system(Command.c_str());
        Command="echo pandoon > " + GPU_path + "/governor";
	system(Command.c_str());

        Command= "echo 1 > " + IOCTL_path;
        return 0;
}
static
int init_rockpi(){
        std::string CPU_path="/sys/devices/system/cpu/cpufreq/";
        std::string GPU_path="/sys/class/devfreq/ff9a0000.gpu/";
        std::string IOCTL_path="/dev/pandoon_device";
        std::string Command="chmod 666 " + IOCTL_path;
        Command="echo pandoon > " + CPU_path + "policy4/scaling_governor";
        system(Command.c_str());
        Command="echo pandoon > " + CPU_path + "policy0/scaling_governor";
        system(Command.c_str());
        Command="echo pandoon > " + GPU_path + "/governor";
	system(Command.c_str());

        Command="cat " + CPU_path + "policy4/scaling_governor";
        system(Command.c_str());
        Command="cat " + CPU_path + "policy4/scaling_governor";
        system(Command.c_str());
        Command="cat " + GPU_path + "/governor";
        system(Command.c_str());
        return 0;
}
static
void read_freqs(){
        std::string CPU_path="/sys/devices/system/cpu/cpufreq/";
        std::string GPU_path="/sys/class/devfreq/ff9a0000.gpu/";
        std::string t="scaling_";
        t="cpuinfo_";
        //You can also open it and read/write it (using open and read c++ commands)
        //Big CPU:
        std::string Command="cat " + CPU_path + "policy0/" + t+"cur_freq";
        system(Command.c_str());
        //Little CPU
        Command="cat " + CPU_path + "policy4/" + t+"cur_freq";
        system(Command.c_str());
        //GPU:
        Command="cat " + GPU_path + "/cur_freq";
        system(Command.c_str());

}
static
int m(){
        init_rockpi();
        int fd=open_pandoon();
        std::string s;
        /*while(true){
                std::cout<<"Please enter freqs: ";
                std::cin>>s;
                set_freq(fd,s);
                usleep(2500);
                get_freq(fd);
                read_freqs();
        }*/
        while(true){
                int l,b,g;
                std::cout<<"Please enter freqs: ";
                std::cin>>l;
                std::cin>>b;
                std::cin>>g;
                set_freq(fd,l,b,g);
                usleep(4500);
                get_freq(fd);
                read_freqs();
        }
        return 0;
}
