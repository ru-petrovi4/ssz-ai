using Ssz.AI.Core.Grafana;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace Ssz.AI.Helpers
{
    public static class HWDDHelper
    {
        #region public functions

        public const int HWDDImageWidthPixels = 500;

        public const int HWDDImageHeightPixels = 500;

        //public static (byte[] labels, byte[][] images) ReadHWDD(string labelsPath, string imagesPath)
        //{
        //    // Чтение меток
        //    byte[] labels = ReadLabels(labelsPath);

        //    // Чтение изображений
        //    byte[][] images = ReadImages(imagesPath);

        //    return (labels, images);
        //}

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

        #endregion

        #region private functions
        

        #endregion       
    }
}
