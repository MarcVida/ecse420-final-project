#ifndef BICUBIC_H
#define BICUBIC_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* bicubic_interpolate(const unsigned char* input,
                                   unsigned in_width,
                                   unsigned in_height,
                                   unsigned *out_width,
                                   unsigned *out_height);

#ifdef __cplusplus
}
#endif

#endif 
