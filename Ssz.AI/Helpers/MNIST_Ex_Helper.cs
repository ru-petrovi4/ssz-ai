using Avalonia;
using System;
using System.Drawing;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
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

            PixelSize imagesSize = new(500, 500);

            // Чтение изображений
            byte[][] imageDatas = ReadImageDatas(imagesPath, labels.Length, imagesSize);

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

        private static byte[][] ReadImageDatas(string path, int imagesCount, PixelSize imagesSize)
        {
            byte[][] imageDatas = new byte[imagesCount][];

            int imageSize = imagesSize.Width * imagesSize.Height;
            using (var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open))
            {
                using (var accessor = mmf.CreateViewAccessor())
                {
                    long dataOffset = SkipNpyHeader(accessor);
                    
                    for (long i = 0; i < imagesCount; i += 1)
                    {
                        byte[] imageData = new byte[imageSize];

                        long imagePos = dataOffset + i * imageSize;
                        accessor.ReadArray(imagePos, imageData, 0, imageSize);

                        for (int j = 0; j < imageData.Length; j += 1)
                        {
                            imageData[j] = (byte)(255 - imageData[j]); // Inverse brightness
                        }

                        imageDatas[i] = imageData;
                    }
                }
            }            
            
            return imageDatas;
        }

        #endregion

        private static long SkipNpyHeader(MemoryMappedViewAccessor accessor)
        {
            long offset = 0;

            // 1. Чтение magic string (6 байт)
            byte[] magic = new byte[6];
            accessor.ReadArray(offset, magic, 0, 6);
            offset += 6;

            string magicStr = Encoding.ASCII.GetString(magic);
            if (magicStr.Substring(1) != "NUMPY")
                throw new InvalidDataException("Не является файлом .npy");

            // 2. Версия
            byte major = accessor.ReadByte(offset++);
            byte minor = accessor.ReadByte(offset++);

            // 3. Длина заголовка
            int headerLen;
            if (major == 1 || major == 2)
            {
                headerLen = accessor.ReadUInt16(offset);
                offset += 2;
            }
            else
            {
                headerLen = accessor.ReadInt32(offset);
                offset += 4;
            }

            // 4. Чтение заголовка
            byte[] headerBytes = new byte[headerLen];
            accessor.ReadArray(offset, headerBytes, 0, headerLen);
            offset += headerLen;

            string header = Encoding.ASCII.GetString(headerBytes);
            if (!header.Contains("descr") || !header.Contains("shape"))
                throw new InvalidDataException("Заголовок не содержит ожидаемых полей");

            return offset; // Позиция, где начинаются бинарные данные
        }
    }
}
