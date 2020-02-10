#include <ConsumerImplHelper/ToFCamera.h>
#include <opencv2/opencv.hpp>

using namespace GenTLConsumerImplHelper;
using namespace std;
using namespace cv;


void pointCloudMap(const PartInfo& pointCloud, float* xMap, float* yMap, float* zMap)
{
    const int width = (int)pointCloud.width;
    const int height = (int)pointCloud.height;
    CToFCamera::Coord3D *pPoint = reinterpret_cast<CToFCamera::Coord3D*>(pointCloud.pData);

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col, ++pPoint, ++xMap, ++yMap, ++zMap)
        {
            if (pPoint->IsValid())
            {
                *xMap = pPoint->x;
                *yMap = pPoint->y;
                *zMap = pPoint->z;
            }
            else
            {
                // No depth information available for this pixel. Zero it.
                *xMap = 0;
                *yMap = 0;
                *zMap = 0;
            }
        }
    }
}

class Sample
{
public:
    int run();

protected:
    bool onImageGrabbed(GrabResult grabResult, BufferParts);

private:
    CToFCamera      m_Camera;   // the ToF camera
    int             m_minDepth; // minimum distance value [mm]
    int             m_maxDepth; // maximum distance value [mm]
};

int Sample::run()
{
    try
    {
        m_Camera.OpenFirstCamera();

        // Enable 3D (point cloud) data and intensity data.
        GenApi::CEnumerationPtr ptrComponentSelector = m_Camera.GetParameter("ComponentSelector");
        GenApi::CBooleanPtr ptrComponentEnable = m_Camera.GetParameter("ComponentEnable");
        GenApi::CEnumerationPtr ptrPixelFormat = m_Camera.GetParameter("PixelFormat");
        ptrComponentSelector->FromString("Range");
        ptrComponentEnable->SetValue(true);
        ptrPixelFormat->FromString("Coord3D_ABC32f");
        ptrComponentSelector->FromString("Intensity");
        ptrComponentEnable->SetValue(true);

        // Query the minimum and maximum distance parameters.
        m_minDepth = (int) GenApi::CIntegerPtr(m_Camera.GetParameter("DepthMin"))->GetValue();
        m_maxDepth = (int) GenApi::CIntegerPtr(m_Camera.GetParameter("DepthMax"))->GetValue();

        // Acquire and process images until the call-back onImageGrabbed indicates to stop acquisition. 
        m_Camera.GrabContinuous(2, 1000, this, &Sample::onImageGrabbed);

        // Clean-up
        m_Camera.Close();
    }
    catch (const GenICam::GenericException& e)
    {
        cerr << "Exception occurred: " << e.GetDescription() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Called for each frame grabbed.
bool Sample::onImageGrabbed(GrabResult grabResult, BufferParts parts)
{
    int key = 0;
    if (grabResult.status == GrabResult::Timeout)
    {
        cerr << "Timeout occurred. Acquisition stopped." << endl;
        return false; // Indicate to stop acquisition
    }
    if (grabResult.status != GrabResult::Ok)
    {
        cerr << "Image was not grabbed." << endl;
    }
    else
    {
        try
        {
            // -------  Create OpenCV images from grabbed buffer

            // First the intensity image...
            const int width = (int)parts[0].width;
            const int height = (int)parts[0].height;
            const int count = width * height;

            // ... then the range images.
            float* xMap = new float[count];
            float* yMap = new float[count];
            float* zMap = new float[count];
            pointCloudMap(parts[0], xMap, yMap, zMap);

            // -------   send the images
            std::stringstream is;
            std::vector<unsigned char> bufferx(count*4);
            std::vector<unsigned char> buffery(count*4);
            std::vector<unsigned char> bufferz(count*4);
            std::vector<unsigned char> buffer0(count*2);
            // std::vector<unsigned char> buffer1(count*2);
            memcpy(&bufferx[0], xMap, count*4);
            memcpy(&buffery[0], yMap, count*4);
            memcpy(&bufferz[0], zMap, count*4);
            memcpy(&buffer0[0], parts[1].pData, count*2);
            // memcpy(&buffer1[0], parts[2].pData, count*2);
            is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&bufferx[0]), bufferx.size());
            std::cout << is.rdbuf();
            is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffery[0]), buffery.size());
            std::cout << is.rdbuf();
            is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&bufferz[0]), bufferz.size());
            std::cout << is.rdbuf();
            is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer0[0]), buffer0.size());
            std::cout << is.rdbuf();
            // is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&buffer1[0]), buffer1.size());
            // std::cout << is.rdbuf();

            // ------ Since rangeMap and rangeMapColor didn't take ownership of the memory, 
            //        we must free the memory to prevent memory leaks.
            delete[] xMap;
            delete[] yMap;
            delete[] zMap;
        }
        catch (const Exception&  e)
        {
            cerr << e.what() << endl;
        }
    }
    key = waitKey(1);
    return 'q' != (char)key;
}

int main(int argc, char* argv[])
{
    int exitCode = EXIT_SUCCESS;
    try
    {
        CToFCamera::InitProducer();
        Sample processing;
        exitCode = processing.run();
    }
    catch (GenICam::GenericException& e)
    {
        cerr << "Exception occurred: " << endl << e.GetDescription() << endl;
        exitCode = EXIT_FAILURE;
    }

    // Release the GenTL producer and all of its resources. 
    // Note: Don't call TerminateProducer() until the destructor of the CToFCamera
    // class has been called. The destructor may require resources which may not
    // be available anymore after TerminateProducer() has been called.
    if (CToFCamera::IsProducerInitialized())
        CToFCamera::TerminateProducer();  // Won't throw any exceptions

    return exitCode;
}
