using Ssz.AI.Core.Grafana;
using System;
using System.Collections.Generic;
using System.IO;

namespace Ssz.AI.Helpers
{
    public static class MNISTReader
    {
        public static (byte[] labels, byte[][] images) ReadMNIST(string labelsPath, string imagesPath)
        {
            // Чтение меток
            byte[] labels = ReadLabels(labelsPath);

            // Чтение изображений
            byte[][] images = ReadImages(imagesPath);

            return (labels, images);
        }

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
    }
}
