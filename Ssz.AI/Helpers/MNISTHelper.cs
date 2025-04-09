using Ssz.AI.Core.Grafana;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace Ssz.AI.Helpers
{
    public static class MNISTHelper
    {
        #region public functions

        public const int MNISTImageWidthPixels = 28;

        public const int MNISTImageHeightPixels = 28;

        public static (byte[] labels, byte[][] images) ReadMNIST(string labelsPath, string imagesPath)
        {
            // Чтение меток
            byte[] labels = ReadLabels(labelsPath);

            // Чтение изображений
            byte[][] images = ReadImages(imagesPath);

            return (labels, images);
        }

        public static Bitmap GetBitmap(byte[] mnistImageData, int width, int height)
        {
            Bitmap bitmap = new Bitmap(width, height);

            // Проходим по каждому пикселю и устанавливаем его в Bitmap
            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    // Значение пикселя из массива байтов
                    byte pixelValue = mnistImageData[x + y * width];

                    // Преобразуем значение пикселя в оттенок серого (0-255)
                    Color color = Color.FromArgb(pixelValue, pixelValue, pixelValue);

                    // Устанавливаем пиксель в изображении
                    bitmap.SetPixel(x, y, color);
                }
            }

            return bitmap;
        }

        public static byte GetInterpolatedValue(byte[] mnistImage, float centerX, float centerY)
        {
            int x = (int)centerX;
            int y = (int)centerY;
            if (x < 0 ||
                    y < 0 ||
                    x + 1 >= MNISTImageWidthPixels ||
                    y + 1 >= MNISTImageHeightPixels)
                return 0;

            double tx = centerX - x;
            double ty = centerY - y;
            double interpolatedValue = (1 - tx) * (1 - ty) * mnistImage[x + MNISTImageWidthPixels * y] +
                tx * (1 - ty) * mnistImage[x + 1 + MNISTImageWidthPixels * y] +
                (1 - tx) * ty * mnistImage[x + MNISTImageWidthPixels * (y + 1)] +
                tx * ty * mnistImage[x + 1 + MNISTImageWidthPixels * (y + 1)];            

            return (byte)interpolatedValue;
        }        

        #endregion

        #region private functions

        private static byte[] ReadLabels(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                // Чтение заголовка меток (8 байт)
                int magicNumber = ReadInt32BigEndian(br);
                int numLabels = ReadInt32BigEndian(br);

                // Чтение самих меток
                byte[] labels = br.ReadBytes(numLabels);

                return labels;
            }
        }

        private static byte[][] ReadImages(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                // Чтение заголовка изображений (16 байт)
                int magicNumber = ReadInt32BigEndian(br);
                int numImages = ReadInt32BigEndian(br);
                int numRows = ReadInt32BigEndian(br);
                int numCols = ReadInt32BigEndian(br);

                // Создаем массив для хранения изображений
                byte[][] images = new byte[numImages][];

                // Чтение пиксельных данных
                for (int i = 0; i < numImages; i += 1)
                {
                    byte[] image = br.ReadBytes(numRows * numCols);
                    images[i] = image;
                }

                return images;
            }
        }

        private static int ReadInt32BigEndian(BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(4);
            Array.Reverse(bytes); // Переворачиваем байты для Big-Endian
            return BitConverter.ToInt32(bytes, 0);
        }

        #endregion       
    }
}
