#include "HOG.h"
#include "CudaChecks.hpp"

#define MAX_THREADS_1D 256
#define MAX_THREADS_2D  16

void ZhuRamananHOGdescriptor_cpu(double *inputImage,
    int cellHeightAndWidthInPixels,
    unsigned int imageHeight,
    unsigned int imageWidth,
    unsigned int numberOfChannels,
    double *descriptorMatrix);
void DalalTriggsHOGdescriptor_cpu(double *inputImage,
    unsigned int numberOfOrientationBins,
    unsigned int cellHeightAndWidthInPixels,
    unsigned int blockHeightAndWidthInCells,
    bool signedOrUnsignedGradientsBool,
    double l2normClipping,
    unsigned int imageHeight,
    unsigned int imageWidth,
    unsigned int numberOfChannels,
    double *descriptorVector);

#ifdef __CUDACC__
    #define CUDA_MAJOR 2
    #define CUDA_MINOR 0
    int __DEVICE_COUNT__(-1); // -1 means to be done
    bool is_cuda_available();
    __device__ double atomicAdd(double *address, double val);
    
    // ZhuRamananHOG
    __global__ void kernel_image_ZhuRamananHOGdescriptor(double *d_hist,
        const dim3 visible,
        const dim3 blocks,
        const double *d_inputImage,
        const unsigned int imageHeight,
        const unsigned int imageWidth,
        const unsigned int numberOfChannels,
        const int cellHeightAndWidthInPixels);
    void ZhuRamananHOGdescriptor_cuda(double *inputImage,
        int cellHeightAndWidthInPixels,
        unsigned int imageHeight,
        unsigned int imageWidth,
        unsigned int numberOfChannels,
        double *descriptorMatrix);
    
    // DalalTriggsHOG
    __global__ void kernel_image_DalalTriggsHOGdescriptor(double *d_h,
        const dim3 h_dims,
        const double *d_inputImage,
        const unsigned int imageHeight,
        const unsigned int imageWidth,
        const unsigned int numberOfChannels,
        const unsigned int numberOfOrientationBins,
        const unsigned int cellHeightAndWidthInPixels,
        const unsigned signedOrUnsignedGradients,
        const double binsSize);
    void DalalTriggsHOGdescriptor_cuda(double *inputImage,
        unsigned int numberOfOrientationBins,
        unsigned int cellHeightAndWidthInPixels,
        unsigned int blockHeightAndWidthInCells,
        bool signedOrUnsignedGradientsBool,
        double l2normClipping,
        unsigned int imageHeight,
        unsigned int imageWidth,
        unsigned int numberOfChannels,
        double *descriptorVector);
#endif

/**
    HOG methods
*/

HOG::HOG(unsigned int windowHeight, unsigned int windowWidth,
         unsigned int numberOfChannels, unsigned int method,
         unsigned int numberOfOrientationBins,
         unsigned int cellHeightAndWidthInPixels,
         unsigned int blockHeightAndWidthInCells, bool enableSignedGradients,
         double l2normClipping) {
    unsigned int descriptorLengthPerBlock = 0,
                 numberOfBlocksPerWindowVertically = 0,
                 numberOfBlocksPerWindowHorizontally = 0;

    if (method == 1) {
        descriptorLengthPerBlock = blockHeightAndWidthInCells *
                                   blockHeightAndWidthInCells *
                                   numberOfOrientationBins;
        numberOfBlocksPerWindowVertically = 1 +
        (windowHeight - blockHeightAndWidthInCells*cellHeightAndWidthInPixels)
        / cellHeightAndWidthInPixels;
        numberOfBlocksPerWindowHorizontally = 1 +
        (windowWidth - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
        / cellHeightAndWidthInPixels;
    }
    else if (method==2) {
        descriptorLengthPerBlock = 27 + 4;
        numberOfBlocksPerWindowVertically =
        (unsigned int)round((double)windowHeight /
                            (double)cellHeightAndWidthInPixels) - 2;
        numberOfBlocksPerWindowHorizontally =
        (unsigned int)round((double)windowWidth /
                            (double)cellHeightAndWidthInPixels) - 2;
    }
    this->method = method;
    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
    this->enableSignedGradients = enableSignedGradients;
    this->l2normClipping = l2normClipping;
    this->numberOfBlocksPerWindowHorizontally =
                    numberOfBlocksPerWindowHorizontally;
    this->numberOfBlocksPerWindowVertically =
                    numberOfBlocksPerWindowVertically;
    this->descriptorLengthPerBlock = descriptorLengthPerBlock;
    this->descriptorLengthPerWindow = numberOfBlocksPerWindowHorizontally *
                                      numberOfBlocksPerWindowVertically *
                                      descriptorLengthPerBlock;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

HOG::~HOG() {}

void HOG::apply(double *windowImage, double *descriptorVector)
{
    #ifdef __CUDACC__
        if (is_cuda_available())
        {
            if (this->method == 1)
                DalalTriggsHOGdescriptor_cuda(windowImage, this->numberOfOrientationBins,
                    this->cellHeightAndWidthInPixels,
                    this->blockHeightAndWidthInCells,
                    this->enableSignedGradients,
                    this->l2normClipping, this->windowHeight,
                    this->windowWidth, this->numberOfChannels,
                    descriptorVector);
            else
                ZhuRamananHOGdescriptor_cuda(windowImage, this->cellHeightAndWidthInPixels,
                    this->windowHeight, this->windowWidth,
                    this->numberOfChannels, descriptorVector);
        }
        else
        {
            if (this->method == 1)
                DalalTriggsHOGdescriptor_cpu(windowImage, this->numberOfOrientationBins,
                    this->cellHeightAndWidthInPixels,
                    this->blockHeightAndWidthInCells,
                    this->enableSignedGradients,
                    this->l2normClipping, this->windowHeight,
                    this->windowWidth, this->numberOfChannels,
                    descriptorVector);
            else
                ZhuRamananHOGdescriptor_cpu(windowImage, this->cellHeightAndWidthInPixels,
                    this->windowHeight, this->windowWidth,
                    this->numberOfChannels, descriptorVector);
        }
    #else
        if (this->method == 1)
            DalalTriggsHOGdescriptor_cpu(windowImage, this->numberOfOrientationBins,
                this->cellHeightAndWidthInPixels,
                this->blockHeightAndWidthInCells,
                this->enableSignedGradients,
                this->l2normClipping, this->windowHeight,
                this->windowWidth, this->numberOfChannels,
                descriptorVector);
        else
            ZhuRamananHOGdescriptor_cpu(windowImage, this->cellHeightAndWidthInPixels,
                this->windowHeight, this->windowWidth,
                this->numberOfChannels, descriptorVector);
    #endif
}

/**
    CPU code
*/

// ZHU & RAMANAN: Face Detection, Pose Estimation and Landmark Localization
//                in the Wild
void ZhuRamananHOGdescriptor_cpu(double *inputImage,
                                 int cellHeightAndWidthInPixels,
                                 unsigned int imageHeight, unsigned int imageWidth,
                                 unsigned int numberOfChannels,
                                 double *descriptorMatrix) {
    // unit vectors used to compute gradient orientation
    double uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000,
                    -0.7660, -0.9397};
    double vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660,
                    0.6428, 0.3420};
    int x, y, o;

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((double)imageHeight /
                           (double)cellHeightAndWidthInPixels);
    blocks[1] = (int)round((double)imageWidth /
                           (double)cellHeightAndWidthInPixels);
    double *hist = (double *)calloc(blocks[0] * blocks[1] * 18, sizeof(double));
    double *norm = (double *)calloc(blocks[0] * blocks[1], sizeof(double));

    // memory for HOG features
    int out[3];
    out[0] = max(blocks[0]-2, 0);
    out[1] = max(blocks[1]-2, 0);
    out[2] = 27+4;

    int visible[2];
    visible[0] = blocks[0] * cellHeightAndWidthInPixels;
    visible[1] = blocks[1] * cellHeightAndWidthInPixels;

    for (x = 1; x < visible[1] - 1; x++) {
        for (y = 1; y < visible[0] - 1; y++) {
            // compute gradient
            // first channel
            double *s = inputImage + min(x, imageWidth-2) * imageHeight +
                        min(y, imageHeight-2);
            double dy = *(s + 1) - *(s - 1);
            double dx = *(s + imageHeight) - *(s - imageHeight);
            double v = dx * dx + dy * dy;
            // rest of channels
            for (unsigned int z = 1; z < numberOfChannels; z++) {
                s += imageHeight * imageWidth;
                double dy2 = *(s + 1) - *(s - 1);
                double dx2 = *(s + imageHeight) - *(s - imageHeight);
                double v2 = dx2 * dx2 + dy2 * dy2;
                // pick channel with strongest gradient
                if (v2 > v) {
                    v = v2;
                    dx = dx2;
                    dy = dy2;
                }
            }

            // snap to one of 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (o = 0; o < 9; o++) {
                double dot = uu[o] * dx + vv[o] * dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot) {
                    best_dot = - dot;
                    best_o = o + 9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            double xp = ((double)x + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            double yp = ((double)y + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            double vx0 = xp - ixp;
            double vy0 = yp - iyp;
            double vx1 = 1.0 - vx0;
            double vy1 = 1.0 - vy0;
            v = sqrt(v);

            if (ixp >= 0 && iyp >= 0)
                *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx1 * vy1 * v;

            if (ixp+1 < blocks[1] && iyp >= 0)
                *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx0 * vy1 * v;

            if (ixp >= 0 && iyp+1 < blocks[0])
                *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx1 * vy0 * v;

            if (ixp+1 < blocks[1] && iyp+1 < blocks[0])
                *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx0 * vy0 * v;
        }
    }

    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++) {
        double *src1 = hist + o * blocks[0] * blocks[1];
        double *src2 = hist + (o + 9) * blocks[0] * blocks[1];
        double *dst = norm;
        double *end = norm + blocks[1] * blocks[0];
        while (dst < end) {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    for (x = 0; x < out[1]; x++) {
        for (y = 0; y < out[0]; y++) {
            double *dst = descriptorMatrix + x * out[0] + y;
            double *src, *p, n1, n2, n3, n4;

            p = norm + (x + 1) * blocks[0] + y + 1;
            n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + (x + 1) * blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + x * blocks[0] + y + 1;
            n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + x * blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);

            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;

            // contrast-sensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (int o = 0; o < 18; o++) {
                double h1 = min(*src * n1, 0.2);
                double h2 = min(*src * n2, 0.2);
                double h3 = min(*src * n3, 0.2);
                double h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // contrast-insensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (int o = 0; o < 9; o++) {
                double sum = *src + *(src + 9 * blocks[0] * blocks[1]);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0] * out[1];
            *dst = 0.2357 * t2;
            dst += out[0] * out[1];
            *dst = 0.2357 * t3;
            dst += out[0] * out[1];
            *dst = 0.2357 * t4;
        }
    }
    free(hist);
    free(norm);
}


// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor_cpu(double *inputImage,
                                  unsigned int numberOfOrientationBins,
                                  unsigned int cellHeightAndWidthInPixels,
                                  unsigned int blockHeightAndWidthInCells,
                                  bool signedOrUnsignedGradientsBool,
                                  double l2normClipping, unsigned int imageHeight,
                                  unsigned int imageWidth,
                                  unsigned int numberOfChannels,
                                  double *descriptorVector) {
    
    numberOfOrientationBins = (int)numberOfOrientationBins;
    cellHeightAndWidthInPixels = (double)cellHeightAndWidthInPixels;
    blockHeightAndWidthInCells = (int)blockHeightAndWidthInCells;

    unsigned int signedOrUnsignedGradients;
    
    if (signedOrUnsignedGradientsBool) {
        signedOrUnsignedGradients = 1;
    } else {
        signedOrUnsignedGradients = 0;
    }

    int hist1 = 2 + (imageHeight / cellHeightAndWidthInPixels);
    int hist2 = 2 + (imageWidth / cellHeightAndWidthInPixels);

    double binsSize = (1 + (signedOrUnsignedGradients == 1)) *
                      pi / numberOfOrientationBins;

    float *dx = new float[numberOfChannels];
    float *dy = new float[numberOfChannels];
    float gradientOrientation, gradientMagnitude, tempMagnitude, 
          Xc, Yc, Oc, blockNorm;
    int x1 = 0, x2 = 0, y1 = 0, y2 = 0, bin1 = 0, descriptorIndex = 0;
    unsigned int x, y, i, j, k, bin2;

    vector<vector<vector<double> > > h(hist1, vector<vector<double> >
                                       (hist2, vector<double>
                                        (numberOfOrientationBins, 0.0 ) ) );
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );

    //Calculate gradients (zero padding)
    for(unsigned int y = 0; y < imageHeight; y++) {
        for(unsigned int x = 0; x < imageWidth; x++) {
            if (x == 0) {
                for (unsigned int z = 0; z < numberOfChannels; z++)
                    dx[z] = inputImage[y + (x + 1) * imageHeight +
                                       z * imageHeight * imageWidth];
            }
            else {
                if (x == imageWidth - 1) {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dx[z] = -inputImage[y + (x - 1) * imageHeight +
                                            z * imageHeight * imageWidth];
                }
                else {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dx[z] = inputImage[y + (x + 1) * imageHeight +
                                           z * imageHeight * imageWidth] -
                                inputImage[y + (x - 1) * imageHeight +
                                           z * imageHeight * imageWidth];
                }
            }

            if(y == 0) {
                for (unsigned int z = 0; z < numberOfChannels; z++)
                    dy[z] = -inputImage[y + 1 + x * imageHeight +
                                        z * imageHeight * imageWidth];
            }
            else {
                if (y == imageHeight - 1) {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dy[z] = inputImage[y - 1 + x * imageHeight +
                                           z * imageHeight * imageWidth];
                }
                else {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dy[z] = -inputImage[y + 1 + x * imageHeight +
                                            z * imageHeight * imageWidth] +
                                 inputImage[y - 1 + x * imageHeight +
                                            z * imageHeight * imageWidth];
                }
            }

            // choose dominant channel based on magnitude
            gradientMagnitude = sqrt(dx[0] * dx[0] + dy[0] * dy[0]);
            gradientOrientation= atan2(dy[0], dx[0]);
            if (numberOfChannels > 1) {
                tempMagnitude = gradientMagnitude;
                for (unsigned int cli = 1; cli < numberOfChannels; ++cli) {
                    tempMagnitude= sqrt(dx[cli] * dx[cli] + dy[cli] * dy[cli]);
                    if (tempMagnitude > gradientMagnitude) {
                        gradientMagnitude = tempMagnitude;
                        gradientOrientation = atan2(dy[cli], dx[cli]);
                    }
                }
            }

            if (gradientOrientation < 0)
                gradientOrientation += pi +
                                       (signedOrUnsignedGradients == 1) * pi;

            // trilinear interpolation
            bin1 = (gradientOrientation / binsSize) - 1;
            bin2 = bin1 + 1;
            x1   = x / cellHeightAndWidthInPixels;
            x2   = x1 + 1;
            y1   = y / cellHeightAndWidthInPixels;
            y2   = y1 + 1;

            Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
            Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
            Oc = (bin1 + 1 + 1 - 1.5) * binsSize;

            if (bin2 == numberOfOrientationBins)
                bin2 = 0;

            if (bin1 < 0)
                bin1 = numberOfOrientationBins - 1;

            h[y1][x1][bin1] = h[y1][x1][bin1] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y1][x1][bin2] = h[y1][x1][bin2] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin1] = h[y2][x1][bin1] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin2] = h[y2][x1][bin2] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin1] = h[y1][x2][bin1] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin2] = h[y1][x2][bin2] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin1] = h[y2][x2][bin1] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin2] = h[y2][x2][bin2] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
        }
    }

    //Block normalization
    for(x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for (y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            blockNorm = 0;
            for (i = 0; i < blockHeightAndWidthInCells; i++)
                for(j = 0; j < blockHeightAndWidthInCells; j++)
                    for(k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += h[y+i][x+j][k] * h[y+i][x+j][k];

            blockNorm = sqrt(blockNorm);
            for (i = 0; i < blockHeightAndWidthInCells; i++) {
                for(j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0) {
                            block[i][j][k] = h[y+i][x+j][k] / blockNorm;
                            if (block[i][j][k] > l2normClipping)
                                block[i][j][k] = l2normClipping;
                        }
                    }
                }
            }

            blockNorm = 0;
            for (i = 0; i < blockHeightAndWidthInCells; i++)
                for(j = 0; j < blockHeightAndWidthInCells; j++)
                    for(k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += block[i][j][k] * block[i][j][k];

            blockNorm = sqrt(blockNorm);
            for (i = 0; i < blockHeightAndWidthInCells; i++) {
                for(j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0)
                            descriptorVector[descriptorIndex] =
                                block[i][j][k] / blockNorm;
                        else
                            descriptorVector[descriptorIndex] = 0.0;
                        descriptorIndex++;
                    }
                }
            }
        }
    }
    delete[] dx;
    delete[] dy;
}

/**
    CUDA/GPU code
*/

#ifdef __CUDACC__

/**
    Check if CUDA is available or not
    and compatible or not
    The check itself is only done one time.
    Result is then stored in the variable __DEVICE_COUNT__
    
    from https://github.com/dubzzz/cuda-compile-4-cuda-and-noncuda/
*/
bool is_cuda_available()
{
    if (__DEVICE_COUNT__ == -1)
    {
        int deviceCount;
        cudaError_t e = cudaGetDeviceCount(&deviceCount);
        if (e != cudaSuccess)
            __DEVICE_COUNT__ = 0;
        else
        {
            __DEVICE_COUNT__ = 0;
            
            // for each GPU check if it has the required compute capability
            for (int i(0) ; i != deviceCount ; i++)
            {
                cudaDeviceProp prop;
                e = cudaGetDeviceProperties(&prop, i);
                // check compute capability
                if (e == cudaSuccess && (prop.major > CUDA_MAJOR || (prop.major == CUDA_MAJOR && prop.minor >= CUDA_MINOR)))
                {
                    cudaSetDevice(i);
                    __DEVICE_COUNT__++;
                }
            }
        }
    }
    return __DEVICE_COUNT__ != 0;
}

__device__ double atomicAdd(double* address, double val) // http://stackoverflow.com/questions/16882253/cuda-atomicadd-produces-wrong-result
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
}

// ZHU & RAMANAN: Face Detection, Pose Estimation and Landmark Localization
//                in the Wild
__global__ void kernel_image_ZhuRamananHOGdescriptor(double *d_hist,
                                                     const dim3 visible,
                                                     const dim3 blocks,
                                                     const double *d_inputImage,
                                                     const unsigned int imageHeight,
                                                     const unsigned int imageWidth,
                                                     const unsigned int numberOfChannels,
                                                     const int cellHeightAndWidthInPixels)
{
    // Retrieve pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x +1;
    int y = blockIdx.y * blockDim.y + threadIdx.y +1;
     
    // Check if position is inside the image and not on borders
    // full check: if (x < 1 || y < 1 || x >= visible.x -1 || y >= visible.y -1), but x>=1 and y>=1
    if (x >= visible.x -1 || y >= visible.y -1)
        return;
    
    // Usefull variables
    // unit vectors used to compute gradient orientation
    double uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    double vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420};
    
    // compute gradient
    // first channel
    const double *s = d_inputImage + min(x, imageWidth-2) * imageHeight + min(y, imageHeight-2);
    double dy = *(s + 1) - *(s - 1);
    double dx = *(s + imageHeight) - *(s - imageHeight);
    double v = dx * dx + dy * dy;
    // rest of channels
    for (unsigned int z = 1; z < numberOfChannels; z++)
    {
        s += imageHeight * imageWidth;
        double dy2 = *(s + 1) - *(s - 1);
        double dx2 = *(s + imageHeight) - *(s - imageHeight);
        double v2 = dx2 * dx2 + dy2 * dy2;
        // pick channel with strongest gradient
        if (v2 > v) {
            v = v2;
            dx = dx2;
            dy = dy2;
        }
    }

    // snap to one of 18 orientations
    double best_dot = 0;
    int best_o = 0;
    for (int o = 0; o < 9; o++)
    {
        double dot = uu[o] * dx + vv[o] * dy;
        if (dot > best_dot)
        {
            best_dot = dot;
            best_o = o;
        }
        else if (-dot > best_dot)
        {
            best_dot = - dot;
            best_o = o + 9;
        }
    }

    // add to 4 histograms around pixel using linear interpolation
    double xp = ((double) x + 0.5) / (double) cellHeightAndWidthInPixels - 0.5;
    double yp = ((double) y + 0.5) / (double) cellHeightAndWidthInPixels - 0.5;
    int ixp = (int) floor(xp);
    int iyp = (int) floor(yp);
    double vx0 = xp - ixp;
    double vy0 = yp - iyp;
    double vx1 = 1.0 - vx0;
    double vy1 = 1.0 - vy0;
    v = sqrt(v);

    if (ixp >= 0 && iyp >= 0)
        atomicAdd(&d_hist[ixp*blocks.x + iyp + best_o*blocks.x*blocks.y], vx1 * vy1 * v);

    if (ixp+1 < blocks.y && iyp >= 0)
        atomicAdd(&d_hist[(ixp+1)*blocks.x + iyp + best_o*blocks.x*blocks.y], vx0 * vy1 * v);

    if (ixp >= 0 && iyp+1 < blocks.x)
        atomicAdd(&d_hist[ixp*blocks.x + (iyp+1) + best_o*blocks.x*blocks.y], vx1 * vy0 * v);

    if (ixp+1 < blocks.y && iyp+1 < blocks.x)
        atomicAdd(&d_hist[(ixp+1)*blocks.x + (iyp+1) + best_o*blocks.x*blocks.y], vx0 * vy0 * v);
}

void ZhuRamananHOGdescriptor_cuda(double *inputImage,
                                  int cellHeightAndWidthInPixels,
                                  unsigned int imageHeight, unsigned int imageWidth,
                                  unsigned int numberOfChannels,
                                  double *descriptorMatrix)
{
    // Compute histograms
    // memory for caching orientation histograms
    
    dim3 blocks((int) round((double) imageHeight / (double) cellHeightAndWidthInPixels), (int) round((double) imageWidth / (double) cellHeightAndWidthInPixels), 0);
    dim3 visible(blocks.x * cellHeightAndWidthInPixels, blocks.y * cellHeightAndWidthInPixels, 0);
    double *d_hist;
    cudaErrorCheck(cudaMalloc(&d_hist, blocks.x * blocks.y * 18 * sizeof(double)));
    cudaErrorCheck(cudaMemset(d_hist, 0., blocks.x * blocks.y * 18 * sizeof(double)));
    
    double *d_inputImage;
    cudaErrorCheck(cudaMalloc(&d_inputImage, imageHeight * imageWidth * sizeof(double)));
    cudaErrorCheck(cudaMemcpy(d_inputImage, inputImage, imageHeight * imageWidth * sizeof(double), cudaMemcpyHostToDevice));
    
    const dim3 dimBlock(MAX_THREADS_2D, MAX_THREADS_2D, 1);
    const dim3 dimGrid((visible.x -2 + dimBlock.x -1)/dimBlock.x, (visible.y -2 + dimBlock.y -1)/dimBlock.y, 1); // x in [1,visible.x -1] ; y in [1,visible.y -1]
    kernel_image_ZhuRamananHOGdescriptor<<<dimGrid, dimBlock>>>(d_hist, visible, blocks,
                                                                d_inputImage, imageHeight, imageWidth, numberOfChannels,
                                                                cellHeightAndWidthInPixels);
    cudaErrorCheck(cudaFree(d_inputImage));
    
    // memory for caching orientation histograms on cpu
    double hist[blocks.x * blocks.y * 18];
    cudaErrorCheck(cudaMemcpy(hist, d_hist, blocks.x * blocks.y * 18 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaFree(d_hist));
    
    // memory for caching orientation histograms norms
    double *norm = (double *)calloc(blocks.x * blocks.y, sizeof(double));
    
    // memory for HOG features
    int out[3];
    out[0] = max(blocks.x-2, 0);
    out[1] = max(blocks.y-2, 0);
    out[2] = 27+4;

    // compute energy in each block by summing over orientations
    
    for (int o = 0; o < 9; o++) {
        double *src1 = hist + o * blocks.x * blocks.y;
        double *src2 = hist + (o + 9) * blocks.x * blocks.y;
        double *dst = norm;
        double *end = norm + blocks.y * blocks.x;
        while (dst < end) {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    
    for (int x = 0; x < out[1]; x++) {
        for (int y = 0; y < out[0]; y++) {
            double *dst = descriptorMatrix + x * out[0] + y;
            double *src, *p, n1, n2, n3, n4;

            p = norm + (x + 1) * blocks.x + y + 1;
            n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks.x) +
                            *(p + blocks.x + 1) + eps);
            p = norm + (x + 1) * blocks.x + y;
            n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks.x) +
                            *(p + blocks.x + 1) + eps);
            p = norm + x * blocks.x + y + 1;
            n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks.x) +
                            *(p + blocks.x + 1) + eps);
            p = norm + x * blocks.x + y;
            n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks.x) +
                            *(p + blocks.x + 1) + eps);

            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;

            // contrast-sensitive features
            src = hist + (x + 1) * blocks.x + (y + 1);
            for (int o = 0; o < 18; o++) {
                double h1 = min(*src * n1, 0.2);
                double h2 = min(*src * n2, 0.2);
                double h3 = min(*src * n3, 0.2);
                double h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0] * out[1];
                src += blocks.x * blocks.y;
            }

            // contrast-insensitive features
            src = hist + (x + 1) * blocks.x + (y + 1);
            for (int o = 0; o < 9; o++) {
                double sum = *src + *(src + 9 * blocks.x * blocks.y);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0] * out[1];
                src += blocks.x * blocks.y;
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0] * out[1];
            *dst = 0.2357 * t2;
            dst += out[0] * out[1];
            *dst = 0.2357 * t3;
            dst += out[0] * out[1];
            *dst = 0.2357 * t4;
        }
    }
    free(norm);
}

__global__ void kernel_image_DalalTriggsHOGdescriptor(double *d_h,
                                                      const dim3 h_dims,
                                                      const double *d_inputImage,
                                                      const unsigned int imageHeight,
                                                      const unsigned int imageWidth,
                                                      const unsigned int numberOfChannels,
                                                      const unsigned int numberOfOrientationBins,
                                                      const unsigned int cellHeightAndWidthInPixels,
                                                      const unsigned signedOrUnsignedGradients,
                                                      const double binsSize)
{
    // Retrieve pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
     
    // Check if position is inside the image
    if (x >= imageWidth || y >= imageHeight)
        return;
    
    // Compute deltas
    double dx[3], dy[3];
    
    if (x == 0)
    {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dx[z] = d_inputImage[y + (x + 1) * imageHeight + z * imageHeight * imageWidth];
    }
    else
    {
        if (x == imageWidth - 1)
        {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = -d_inputImage[y + (x - 1) * imageHeight + z * imageHeight * imageWidth];
        }
        else
        {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = d_inputImage[y + (x + 1) * imageHeight + z * imageHeight * imageWidth] - d_inputImage[y + (x - 1) * imageHeight + z * imageHeight * imageWidth];
        }
    }

    if(y == 0)
    {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dy[z] = -d_inputImage[y + 1 + x * imageHeight + z * imageHeight * imageWidth];
    }
    else
    {
        if (y == imageHeight - 1)
        {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = d_inputImage[y - 1 + x * imageHeight + z * imageHeight * imageWidth];
        }
        else
        {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = -d_inputImage[y + 1 + x * imageHeight + z * imageHeight * imageWidth] + d_inputImage[y - 1 + x * imageHeight + z * imageHeight * imageWidth];
        }
    }

    // Choose dominant channel based on magnitude
    double gradientMagnitude = sqrt(dx[0] * dx[0] + dy[0] * dy[0]);
    double gradientOrientation = atan2(dy[0], dx[0]);
    if (numberOfChannels > 1)
    {
        double tempMagnitude = gradientMagnitude;
        for (unsigned int cli = 1 ; cli < numberOfChannels ; ++cli)
        {
            tempMagnitude= sqrt(dx[cli] * dx[cli] + dy[cli] * dy[cli]);
            if (tempMagnitude > gradientMagnitude)
            {
                gradientMagnitude = tempMagnitude;
                gradientOrientation = atan2(dy[cli], dx[cli]);
            }
        }
    }

    if (gradientOrientation < 0)
        gradientOrientation += pi + (signedOrUnsignedGradients == 1) * pi;

    // Trilinear interpolation
    int bin1 = (gradientOrientation / binsSize) - 1;
    unsigned int bin2 = bin1 + 1;
    int x1   = x / cellHeightAndWidthInPixels;
    int x2   = x1 + 1;
    int y1   = y / cellHeightAndWidthInPixels;
    int y2   = y1 + 1;
    
    double Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Oc = (bin1 + 1 + 1 - 1.5) * binsSize;
    
    if (bin2 == numberOfOrientationBins)
        bin2 = 0;
    
    if (bin1 < 0)
        bin1 = numberOfOrientationBins - 1;
    
    atomicAdd(&d_h[y1 + x1 * factor_y_dim + bin1 * factor_z_dim], gradientMagnitude *
                                                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y1 + x1 * factor_y_dim + bin2 * factor_z_dim], gradientMagnitude *
                                                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y2 + x1 * factor_y_dim + bin1 * factor_z_dim], gradientMagnitude *
                                                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y2 + x1 * factor_y_dim + bin2 * factor_z_dim], gradientMagnitude *
                                                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y1 + x2 * factor_y_dim + bin1 * factor_z_dim], gradientMagnitude *
                                                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y1 + x2 * factor_y_dim + bin2 * factor_z_dim], gradientMagnitude *
                                                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y2 + x2 * factor_y_dim + bin1 * factor_z_dim], gradientMagnitude *
                                                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(&d_h[y2 + x2 * factor_y_dim + bin2 * factor_z_dim], gradientMagnitude *
                                                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                                  (((gradientOrientation-Oc)/binsSize)));
}

// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor_cuda(double *inputImage,
                                   unsigned int numberOfOrientationBins,
                                   unsigned int cellHeightAndWidthInPixels,
                                   unsigned int blockHeightAndWidthInCells,
                                   bool signedOrUnsignedGradientsBool,
                                   double l2normClipping, unsigned int imageHeight,
                                   unsigned int imageWidth,
                                   unsigned int numberOfChannels,
                                   double *descriptorVector) {
    
    //Calculate gradients (zero padding)
    //using CUDA
    
    const int hist1 = 2 + (imageHeight / cellHeightAndWidthInPixels);
    const int hist2 = 2 + (imageWidth / cellHeightAndWidthInPixels);
    const dim3 h_dims(hist1, hist2, numberOfOrientationBins);
    const unsigned int factor_z_dim = h_dims.x * h_dims.y;
    const unsigned int factor_y_dim = h_dims.x;
    double *d_h;
    cudaErrorCheck(cudaMalloc(&d_h, h_dims.x * h_dims.y * h_dims.z * sizeof(double)));
    cudaErrorCheck(cudaMemset(d_h, 0., h_dims.x * h_dims.y * h_dims.z * sizeof(double)));
    
    double *d_inputImage;
    cudaErrorCheck(cudaMalloc(&d_inputImage, imageHeight * imageWidth * sizeof(double)));
    cudaErrorCheck(cudaMemcpy(d_inputImage, inputImage, imageHeight * imageWidth * sizeof(double), cudaMemcpyHostToDevice));
    
    const dim3 dimBlock(MAX_THREADS_2D, MAX_THREADS_2D, 1);
    const dim3 dimGrid((imageWidth + dimBlock.x -1)/dimBlock.x, (imageHeight + dimBlock.y -1)/dimBlock.y, 1);
    kernel_image_DalalTriggsHOGdescriptor<<<dimGrid, dimBlock>>>(d_h, h_dims,
                                                                 d_inputImage, imageHeight, imageWidth, numberOfChannels,
                                                                 numberOfOrientationBins, cellHeightAndWidthInPixels,
                                                                 signedOrUnsignedGradientsBool ? 1 : 0 /*signedOrUnsignedGradients*/,
                                                                 (1 + (signedOrUnsignedGradientsBool ? 1 : 0)) * pi / numberOfOrientationBins /*binsSize*/);
    cudaThreadSynchronize(); // block until the device is finished
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        cudaErrorRaise(error);
    
    double h[h_dims.x * h_dims.y * h_dims.z];
    cudaErrorCheck(cudaMemcpy(h, d_h, h_dims.x * h_dims.y * h_dims.z * sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaErrorCheck(cudaFree(d_h));
    cudaErrorCheck(cudaFree(d_inputImage));
    
    //Block normalization
    
    int descriptorIndex(0);
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );
    
    for (unsigned int x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for (unsigned int y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            float blockNorm(0);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++)
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++)
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += h[y+i + (x+j) * factor_y_dim + k * factor_z_dim] * h[y+i + (x+j) * factor_y_dim + k * factor_z_dim];

            blockNorm = sqrt(blockNorm);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++) {
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++) {
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0) {
                            block[i][j][k] = h[y+i + (x+j) * factor_y_dim + k * factor_z_dim] / blockNorm;
                            if (block[i][j][k] > l2normClipping)
                                block[i][j][k] = l2normClipping;
                        }
                    }
                }
            }

            blockNorm = 0;
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++)
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++)
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += block[i][j][k] * block[i][j][k];

            blockNorm = sqrt(blockNorm);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++) {
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++) {
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++, descriptorIndex++) {
                        if (blockNorm > 0)
                            descriptorVector[descriptorIndex] = block[i][j][k] / blockNorm;
                        else
                            descriptorVector[descriptorIndex] = 0.0;
                    }
                }
            }
        }
    }
}

#endif

