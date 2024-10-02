#include "../imread.h"

void toGray2D(
    unsigned char* I_h,
    unsigned char* O_h,
    unsigned long w,
    unsigned long h) {

    }

int main() {
    unsigned char * img;
    unsigned char * out;
    unsigned long x, y, size;

    tie(img, x, y) = imRead();
    
    size = x * y;
    out = (unsigned char*)malloc(size);

    toGray2D(img, out, x, y);

    printf("read image! %lu %lu", x, y);

    free(img);
    free(out);
    return 0;
}