using OpenCvSharp;
using System;
using System.DrawingCore;
using System.DrawingCore.Imaging;
using System.IO;

namespace Ssz.AI.Helpers
{
    public static class BitmapHelper
    {
        public static Avalonia.Media.Imaging.Bitmap ConvertMatToBitmap(Mat image)
        {
            using (var memoryStream = new MemoryStream())
            {
                var bitmap = ToBitmap(image);
                bitmap.Save(memoryStream, ImageFormat.Png);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return new Avalonia.Media.Imaging.Bitmap(memoryStream);
            }
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
