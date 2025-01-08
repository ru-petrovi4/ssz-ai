using Avalonia.Layout;
using Microsoft.AspNetCore.DataProtection.XmlEncryption;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using static Ssz.AI.Models.Cortex;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model4
    {
        #region construction and destruction

        /// <summary>
        ///     Построение графика распределения величин градиентов
        /// </summary>
        public Model4()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                gradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            _retina = new Retina(Constants, gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);

            // Вызываем для вычисления начального вектора активации детекторов
            GetImages(0.0, 0.0);

            Cortex = new Cortex(Constants, _retina);

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            foreach (var gradientMatrix in gradientMatricesCollection)
            {
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: _retina.Detectors.Dimensions[1],
                    dy =>
                    {
                        foreach (int dx in Enumerable.Range(0, _retina.Detectors.Dimensions[0]))
                        {
                            var d = _retina.Detectors[dx, dy];
                            d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                        }                            
                    });               

                foreach (var miniColumn in Cortex.MiniColumns.Data)
                {
                    miniColumn.CalculateHash(miniColumn.Temp_Hash);
                    int bitsCountInHash = (int)TensorPrimitives.Sum(miniColumn.Temp_Hash);
                    //dataToDisplayHolder.MiniColumsActivatedDetectorsCountDistribution[activatedDetectors.Intersect(miniColumn.Detectors).Count()] += 1;
                    dataToDisplayHolder.MiniColumsBitsCountInHashDistribution[bitsCountInHash] += 1;
                }                
            }
        }

        #endregion

        #region public functions       

        public readonly ModelConstants Constants = new();

        public double DetectorsActivationScalarProduct0 { get; set; }
        public double DetectorsActivationScalarProduct { get; set; }

        public int CenterX { get; set; }
        public int CenterXDelta { get; set; }
        public int CenterY { get; set; }
        public double AngleDelta { get; set; }
        public double Angle { get; set; }

        public Cortex Cortex { get; }

        public Image[] GetImages(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0); 
            CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + CenterXDelta;
            CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            AngleDelta = angleK * 2.0 * Math.PI;
            Angle = Math.PI / 2 + AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(CenterX + lineLength * Math.Cos(Angle));
            int endY = (int)(CenterY + lineLength * Math.Sin(Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(CenterX - lineLength * Math.Cos(Angle));
            int startY = (int)(CenterY - lineLength * Math.Sin(Angle));

            Bitmap originalBitmap = new Bitmap(Constants.GeneratedImageWidth, Constants.GeneratedImageHeight);
            using (Graphics g = Graphics.FromImage(originalBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Создаем кисть и устанавливаем толщину линии
                using (Pen pen = new Pen(Color.White, 15))
                {                    
                    // Рисуем наклонную линию
                    g.DrawLine(pen, startX, startY, endX, endY);
                }
            }

            // Уменьшаем изображение до размера 28x28

            // Создаем пустое изображение 28x28
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            using (Graphics g = Graphics.FromImage(resizedBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры для уменьшения
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Масштабируем изображение
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            List<Detector> activatedDetectors = new List<Detector>(_retina.Detectors.Dimensions[0] * _retina.Detectors.Dimensions[1]);
            foreach (int dy in Enumerable.Range(0, _retina.Detectors.Dimensions[1]))
                foreach (int dx in Enumerable.Range(0, _retina.Detectors.Dimensions[0]))
                {
                    Detector d = _retina.Detectors[dx, dy];
                    if (d.GetIsActivated(gradientMatrix))
                        activatedDetectors.Add(d);
                }            

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);
            if (positionK == 0.0 && angleK == 0.0)
            {
                _detectorsActivationBitmap0 = detectorsActivationBitmap;
            }

            double detectorsActivationScalarProduct = 0.0;
            for (int y = 0; y < _detectorsActivationBitmap0.Height; y += 1)
            {
                for (int x = 0; x < _detectorsActivationBitmap0.Width; x += 1)
                {
                    var p0 = _detectorsActivationBitmap0.GetPixel(x, y);
                    var p = detectorsActivationBitmap.GetPixel(x, y);
                    if (p0.R > 0 && p.R > 0)
                        detectorsActivationScalarProduct += 1.0;
                }
            }
            DetectorsActivationScalarProduct = detectorsActivationScalarProduct;

            if (positionK == 0.0 && angleK == 0.0)
            {
                DetectorsActivationScalarProduct0 = detectorsActivationScalarProduct;
            }
            
            return [originalBitmap, resizedBitmap, gradientBitmap, detectorsActivationBitmap];
        }

        #endregion        

        #region private fields

        private Retina _retina;
        /// <summary>
        ///     Начальная картина активации детекторов (до смещения).
        /// </summary>
        public Bitmap _detectorsActivationBitmap0 = null!;

        #endregion

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {            
            public int ImageWidth => MNISTHelper.MNISTImageWidth;

            public int ImageHeight => MNISTHelper.MNISTImageHeight;

            public int AngleRangesCount => 4;

            public int MagnitudeRangesCount => 4;

            public int GeneratedImageWidth => 280;
            public int GeneratedImageHeight => 280;

            public int CortexWidth => 200;
            public int CortexHeight => 200;

            /// <summary>
            ///     Расстояние между детекторами по коризонтали и вертикали  
            /// </summary>
            public double DetectorDelta => 0.1;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;

            public int HashLength => 200;

            public int? SubAreaMiniColumnsCount => null;
            public int SubAreaCenter_Cx => 0;
            public int SubAreaCenter_Cy => 0;

            /// <summary>
            ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
            /// </summary>
            public int InitialMemoryBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 7;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int NearestMiniColumnsDelta => 5;            

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Длина короткого хэш-вектора
            /// </summary>
            public int ShortHashLength => 50;

            /// <summary>
            ///     Количество бит в коротком хэш-векторе
            /// </summary>
            public int ShortHashBitsCount => 11;
        }
    }
}
