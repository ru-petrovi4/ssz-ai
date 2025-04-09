using Avalonia;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using Tensorflow.NumPy;

namespace Ssz.AI.Helpers
{
    public static class MNIST_Ex_Helper
    {
        #region public functions

        public static (byte[] labels, byte[][] imageDatas, PixelSize imagesSize) ReadMNISTEx(string labelsPath, string imagesPath)
        {
            // Чтение меток
            byte[] labels = ReadLabels(labelsPath);

            // Чтение изображений
            (byte[][] imageDatas, PixelSize imagesSize) = ReadImages(imagesPath);

            return (labels, imageDatas, imagesSize);
        }

        #endregion

        #region private functions

        private static byte[] ReadLabels(string path)
        {
            NDArray writerInfo = np.load(path);

            NDArray digits = writerInfo[":", "0"]; // Get all rows, column 0
            
            return digits.ToArray<Int32>().Select(it => (byte)it).ToArray();
        }

        private static (byte[][] imageDatas, PixelSize imagesSize) ReadImages(string path)
        {            
            PixelSize imagesSize = new(500, 500);
            NDArray imagesNDArray = np.load(path);
            long imagesCount = imagesNDArray.shape[0];
            byte[][] imageDatas = new byte[imagesCount][];

            for (int i = 0; i < imagesCount; i += 1)
            {
                NDArray imageNDArray = imagesNDArray[i]; // Each image: (500, 500)

                imageDatas[i] = imageNDArray.ToArray<Int32>().Select(it => (byte)it).ToArray();
            }
            
            return (imageDatas, imagesSize);
        }        

        #endregion
    }
}
