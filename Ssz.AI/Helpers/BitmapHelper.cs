using Avalonia;
using OpenCvSharp;
using Ssz.AI.Models;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

namespace Ssz.AI.Helpers
{
    public static class BitmapHelper
    {
        public static byte GetInterpolatedValue(byte[] imageData, PixelSize imageSize, float centerX, float centerY)
        {
            int x = (int)centerX;
            int y = (int)centerY;
            if (x < 0 ||
                    y < 0 ||
                    x + 1 >= imageSize.Width ||
                    y + 1 >= imageSize.Height)
                return 0;

            double tx = centerX - x;
            double ty = centerY - y;
            double interpolatedValue = (1 - tx) * (1 - ty) * imageData[x + imageSize.Width * y] +
                tx * (1 - ty) * imageData[x + 1 + imageSize.Width * y] +
                (1 - tx) * ty * imageData[x + imageSize.Width * (y + 1)] +
                tx * ty * imageData[x + 1 + imageSize.Width * (y + 1)];

            return (byte)interpolatedValue;
        }

        //public static byte GetInterpolatedValue(byte[] image, System.Drawing.Size imageSize, float centerX, float centerY)
        //{
        //    int x = (int)centerX;
        //    int y = (int)centerY;
        //    if (x < 0 ||
        //            y < 0 ||
        //            x + 1 >= image.GetLength(0) ||
        //            y + 1 >= image.GetLength(1))
        //        return 0;

        //    float tx = centerX - x;
        //    float ty = centerY - y;
        //    float interpolatedValue = (1.0f - tx) * (1.0f - ty) * image[x, y] +
        //        tx * (1.0f - ty) * image[x + 1, y] +
        //        (1.0f - tx) * ty * image[x, y + 1] +
        //        tx * ty * image[x + 1, y + 1];

        //    return (byte)interpolatedValue;
        //}

        //public static Bitmap ConvertImageDataToBitmap(byte[] imageData, int width, int height)
        //{
        //    Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format32bppRgb);

        //    // Create a BitmapData and lock all pixels to be written 
        //    BitmapData bitmapData = bitmap.LockBits(
        //                        new Rectangle(0, 0, bitmap.Width, bitmap.Height),
        //                        ImageLockMode.WriteOnly, bitmap.PixelFormat);
        //    // Copy the data from the byte array into BitmapData.Scan0
        //    Marshal.Copy(imageData, 0, bitmapData.Scan0, imageData.Length);

        //    // Unlock the pixels
        //    bitmap.UnlockBits(bitmapData);

        //    return bitmap;
        //}               

        public static Avalonia.Media.Imaging.Bitmap ConvertImageToAvaloniaBitmap(Image image)
        {
            using (var memoryStream = new MemoryStream())
            {
                image.Save(memoryStream, ImageFormat.Png);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return new Avalonia.Media.Imaging.Bitmap(memoryStream);
            }
        }

        public static Avalonia.Media.Imaging.Bitmap ConvertMatToAvaloniaBitmap(Mat image)
        {
            using (var memoryStream = new MemoryStream())
            {
                var bitmap = ToBitmap(image);
                bitmap.Save(memoryStream, ImageFormat.Png);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return new Avalonia.Media.Imaging.Bitmap(memoryStream);
            }
        }

        public static Bitmap GetSubBitmap(Bitmap bitmap, int centerX, int centerY, double radius)
        {
            Bitmap subBitmap = new Bitmap((int)(radius * 2) + 1, (int)(radius * 2) + 1);

            // Проходим по каждому пикселю и устанавливаем его в Bitmap
            for (int y = (int)(centerY - radius); y < (int)(centerY + radius); y += 1)
            {
                for (int x = (int)(centerX - radius); x < (int)(centerX + radius); x += 1)
                {
                    var r = Math.Sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    if (r > radius)
                    {
                        subBitmap.SetPixel(x - (int)(centerX - radius), y - (int)(centerY - radius), Color.Black);
                    }
                    else
                    {                        
                        Color color = bitmap.GetPixel(x, y);

                        // Устанавливаем пиксель в изображении
                        subBitmap.SetPixel(x - (int)(centerX - radius), y - (int)(centerY - radius), color);
                    }
                }
            }

            return subBitmap;
        }

        private static Bitmap ToBitmap(this Mat src)
        {
            var bitmap = new Bitmap(src.Width, src.Height, src.Channels() switch
            {
                1 => PixelFormat.Format8bppIndexed,
                3 => PixelFormat.Format24bppRgb,
                4 => PixelFormat.Format32bppArgb,
                _ => throw new ArgumentException("Number of channels must be 1, 3 or 4.", "src"),
            });

            ToBitmap(src, bitmap);

            return bitmap;
        }

        private static unsafe void ToBitmap(Mat src, Bitmap dst)
        {
            if (src.IsDisposed)
                throw new ArgumentException("The image is disposed.", nameof(src));
            if (src.Depth() != MatType.CV_8U)
                throw new ArgumentException("Depth of the image must be CV_8U");
            //if (src.IsSubmatrix())
            //    throw new ArgumentException("Submatrix is not supported");
            if (src.Width != dst.Width || src.Height != dst.Height)
                throw new ArgumentException("");

            var pf = dst.PixelFormat;

            if (pf == PixelFormat.Format8bppIndexed)
            {
                var plt = dst.Palette;
                for (var x = 0; x < 256; x++)
                {
                    plt.Entries[x] = Color.FromArgb(x, x, x);
                }
                dst.Palette = plt;
            }

            var w = src.Width;
            var h = src.Height;
            var rect = new Rectangle(0, 0, w, h);
            BitmapData? bd = null;

            var submat = src.IsSubmatrix();
            var continuous = src.IsContinuous();

            try
            {
                bd = dst.LockBits(rect, ImageLockMode.WriteOnly, pf);

                var srcData = src.Data;
                var pSrc = (byte*)(srcData.ToPointer());
                var pDst = (byte*)(bd.Scan0.ToPointer());
                var ch = src.Channels();
                var srcStep = (int)src.Step();
                var dstStep = ((src.Width * ch) + 3) / 4 * 4; // 4の倍数に揃える
                var stride = bd.Stride;

                switch (pf)
                {
                    case PixelFormat.Format1bppIndexed:
                        {
                            if (submat)
                                throw new NotImplementedException("submatrix not supported");

                            // BitmapDataは4byte幅だが、IplImageは1byte幅
                            // 手作業で移し替える                 
                            //int offset = stride - (w / 8);
                            int x = 0;
                            byte b = 0;
                            for (var y = 0; y < h; y++)
                            {
                                for (var bytePos = 0; bytePos < stride; bytePos++)
                                {
                                    if (x < w)
                                    {
                                        for (int i = 0; i < 8; i++)
                                        {
                                            var mask = (byte)(0x80 >> i);
                                            if (x < w && pSrc[srcStep * y + x] == 0)
                                                b &= (byte)(mask ^ 0xff);
                                            else
                                                b |= mask;

                                            x++;
                                        }
                                        pDst[bytePos] = b;
                                    }
                                }
                                x = 0;
                                pDst += stride;
                            }
                            break;
                        }

                    case PixelFormat.Format8bppIndexed:
                    case PixelFormat.Format24bppRgb:
                    case PixelFormat.Format32bppArgb:
                        if (srcStep == dstStep && !submat && continuous)
                        {
                            long bytesToCopy = src.DataEnd.ToInt64() - src.Data.ToInt64();
                            Buffer.MemoryCopy(pSrc, pDst, bytesToCopy, bytesToCopy);
                        }
                        else
                        {
                            for (int y = 0; y < h; y++)
                            {
                                long offsetSrc = (y * srcStep);
                                long offsetDst = (y * dstStep);
                                long bytesToCopy = w * ch;
                                // 一列ごとにコピー
                                Buffer.MemoryCopy(pSrc + offsetSrc, pDst + offsetDst, bytesToCopy, bytesToCopy);
                            }
                        }
                        break;

                    default:
                        throw new NotImplementedException();
                }
            }
            finally
            {
                if (bd is not null)
                    dst.UnlockBits(bd);
            }
        }               
    }
}
