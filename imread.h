#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include <tuple>

using namespace std;

tuple<unsigned char*, unsigned long, unsigned long> imRead() {
    struct jpeg_decompress_struct info; //for our jpeg info
    struct jpeg_error_mgr err;          //the error handler
    
    unsigned long x, y; // width and height of image
    unsigned long data_size; // length of the file
    unsigned char* imgdata; // data for the image

    unsigned char * rowptr[1];    // pointer to an array

    FILE *file = fopen("data/lion.jpg","r");

    info.err = jpeg_std_error(& err);     
    jpeg_create_decompress(& info);

    //if the jpeg file doesn't load
    if(!file) {
        fprintf(stderr, "Error reading JPEG file!");
        return { nullptr, 0, 0 };
    }
    
    jpeg_stdio_src(&info, file);
    jpeg_read_header(&info, TRUE);   // read jpeg file header
    
    jpeg_start_decompress(&info);    // decompress the file

    //set width and height
    x = info.output_width;
    y = info.output_height;
    data_size = x * y * 3;

    imgdata = (unsigned char *)malloc(data_size);
    while (info.output_scanline < info.output_height) // loop
    {
        // Enable jpeg_read_scanlines() to fill our jdata array
        rowptr[0] = (unsigned char *)imgdata +  // secret to method
                3* info.output_width * info.output_scanline; 

        jpeg_read_scanlines(&info, rowptr, 1);
    }

    jpeg_finish_decompress(&info);   //finish decompressing
    jpeg_destroy_decompress(&info);
    fclose(file);                    //close the file

    return { imgdata, x, y };
}