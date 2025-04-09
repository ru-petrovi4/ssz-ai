using Avalonia;
using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using static Ssz.AI.Models.Cortex;
using Size = System.Drawing.Size;

namespace Ssz.AI.Models
{
    public class Model9
    {
        #region construction and destruction

        /// <summary>
        ///     Гистограммы для миниколонок
        /// </summary>
        public Model9()
        {
            Logger = ActivatorUtilities.CreateInstance<Logger<Model9>>(Program.Host.Services);
            DataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

            Random random = new();

            LeftEye = CreateEye(pupil: new Vector3DFloat() { X = -Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f });
            RightEye = CreateEye(pupil: new Vector3DFloat() { X = Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f });

            //string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            //string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений
            //(Labels, Images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            (byte[] inputImagesLabels, byte[][] inputImageDatas, PixelSize inputImagesSize) = MNIST_Ex_Helper.ReadMNISTEx(
                labelsPath: @"Data\WriterInfo.npy", 
                imagesPath: @"Data\Images(500x500).npy"
                );

            GradientDistribution leftEye_GradientDistribution = new();
            GradientDistribution rightEye_GradientDistribution = new();

            StereoInput = new StereoInput(inputImagesSize);
            StereoInput.GenerateOwnedData(
                random,
                Constants,
                leftEye_GradientDistribution,
                rightEye_GradientDistribution,
                inputImagesLabels,
                inputImageDatas,
                LeftEye,
                RightEye);
            //Helpers.SerializationHelper.LoadFromFileIfExists("StereoInput.bin", StereoInput, null);
            StereoInput.Prepare();
            //Helpers.SerializationHelper.SaveToFile("StereoInput.bin", StereoInput, null);

            LeftEye.Retina = new Retina(Constants, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            //LeftEye.Retina.GenerateOwnedData(random, Constants, leftEye_GradientDistribution);            
            Helpers.SerializationHelper.LoadFromFileIfExists("LeftEyeRetina.bin", LeftEye.Retina, null);
            LeftEye.Retina.Prepare();
            //Helpers.SerializationHelper.SaveToFile("LeftEyeRetina.bin", LeftEye.Retina, null);

            RightEye.Retina = new Retina(Constants, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            //RightEye.Retina.GenerateOwnedData(random, Constants, rightEye_GradientDistribution);            
            Helpers.SerializationHelper.LoadFromFileIfExists("RightEyeRetina.bin", RightEye.Retina, null);
            RightEye.Retina.Prepare();
            //Helpers.SerializationHelper.SaveToFile("RightEyeRetina.bin", RightEye.Retina, null);

            PreCortex = new PreCortex(Constants, LeftEye, RightEye);
            PreCortex.GenerateOwnedData();            
            //Helpers.SerializationHelper.LoadFromFileIfExists(@"autoencoder.bin", PreCortex, null);
            PreCortex.Prepare();            
            //Helpers.SerializationHelper.SaveToFile(@"PreCortex.bin", PreCortex, null);

            Task.Factory.StartNew(() =>
            {
                PreCortex.Calculate(StereoInput, LeftEye, RightEye);
            }, TaskCreationOptions.LongRunning);
        }

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();        

        public int CurrentInputIndex = 0;

        public StereoInput StereoInput { get; set; } = null!;

        public readonly Eye LeftEye;

        public readonly Eye RightEye;        

        public readonly PreCortex PreCortex = null!;        

        public int Generated_CenterX { get; set; }
        public int Generated_CenterXDelta { get; set; }
        public int Generated_CenterY { get; set; }
        public double Generated_AngleDelta { get; set; }
        public double Generated_Angle { get; set; }        

        public CancellationTokenSource Temp_StopAutoencoderFinding_CancellationTokenSource  { get; set; } = new();

        public ILogger Logger { get; }

        public DataToDisplayHolder DataToDisplayHolder { get; }

        public Image[] GetImages1(double positionK, double angleK)
        {
            byte[] image = StereoInput.StereoInputItems[0].InputImageData;
            Direction imageNormalDirection = new() { XRadians = (float)(positionK - 0.5) * MathF.PI, YRadians = (float)(angleK - 0.5) * MathF.PI };

            var temp_LeftEye_Image = StereoInput.GetEyeImageData(Constants, image, StereoInput.InputImagesSize, imageNormalDirection, LeftEye);
            var temp_RightEye_Image = StereoInput.GetEyeImageData(Constants, image, StereoInput.InputImagesSize, imageNormalDirection, RightEye);

            return [ MNISTHelper.GetBitmap(temp_LeftEye_Image, Constants.EyeImageWidthPixels, Constants.EyeImageHeightPixels),
                MNISTHelper.GetBitmap(temp_RightEye_Image, Constants.EyeImageWidthPixels, Constants.EyeImageHeightPixels) ];

            //return [];
        }        

        public Image[] GetImages2()
        {
            return [ ];
        }        

        public Image[] GetImages3()
        {
            //var image = Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex);

            return [ ];
        }

        //public void CollectMemories_MNIST(int stepsCount)
        //{
        //    DataToDisplayHolder.MiniColumsBitsCountInHashDistribution2 = new ulong[Constants.CortexWidth, Constants.CortexHeight, Constants.HashLength];

        //    foreach (var _ in Enumerable.Range(0, stepsCount))
        //    {
        //        CurrentInputIndex += 1;

        //        var gradientMatrix = LeftEye_GradientMatricesCollection[CurrentInputIndex];

        //        DoStep_CollectMemories_MNIST(gradientMatrix);
        //    }
        //}
        
        #endregion                

        private Eye CreateEye(Vector3DFloat pupil)
        {
            float kX = (float)Constants.EyeImageWidthPixels / (float)MNISTHelper.MNISTImageWidthPixels;
            float kY = (float)Constants.EyeImageHeightPixels / (float)MNISTHelper.MNISTImageHeightPixels;
            Eye eye = new();
            eye.Pupil = pupil;
            eye.Retina = new Retina(Constants, Constants.EyeImageWidthPixels, Constants.EyeImageHeightPixels);
            eye.RetinaUpperLeftXRadians = MathF.Atan2(Constants.ImageCenter.X - kX * Constants.ImageWidth / 2 - pupil.X, Constants.ImageCenter.Z - pupil.Z);
            eye.RetinaUpperLeftYRadians = MathF.Atan2(Constants.ImageCenter.Y - kY * Constants.ImageHeight / 2 - pupil.Y, Constants.ImageCenter.Z - pupil.Z);
            eye.RetinaBottomRightXRadians = MathF.Atan2(Constants.ImageCenter.X + kX * Constants.ImageWidth / 2 - pupil.X, Constants.ImageCenter.Z - pupil.Z);
            eye.RetinaBottomRightYRadians = MathF.Atan2(Constants.ImageCenter.Y + kY * Constants.ImageHeight / 2 - pupil.Y, Constants.ImageCenter.Z - pupil.Z);
            return eye;
        }                              

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {
            /// <summary>
            ///     Расстояние между детекторами по горизонтали и вертикали  
            ///     [0..MNISTImageWidth]
            /// </summary>
            public double DetectorDelta => 0.1;

            public int AngleRangeDegree_LimitMagnitude => 300;

            public double DetectorMinGradientMagnitude => 5;

            public int GeneratedMinGradientMagnitude => 5;

            public int GeneratedMaxGradientMagnitude => 1200;

            public int AngleRangeDegreeMin => 60;

            public int AngleRangeDegreeMax => 60;

            public int MagnitudeRangesCount => 4;

            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidthPixels => MNISTHelper.MNISTImageWidthPixels;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeightPixels => MNISTHelper.MNISTImageHeightPixels;

            public int GeneratedImageWidth => 280;

            public int GeneratedImageHeight => 280;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси X
            /// </summary>
            public int CortexWidth => 200;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси Y
            /// </summary>
            public int CortexHeight => 200;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;

            public int HashLength => 200;

            public int ShortHashLength => 50;

            public int ShortHashBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 11;

            /// <summary>
            ///     Примерное количество воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 400;

            /// <summary>
            ///     Индекс X центра подобласти [0..CortexWidth]
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти [0..CortexHeight]
            /// </summary>
            public int SubAreaCenter_Cy => 100;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int MiniColumnsMaxDistance => 1;

            public float DistanceBetweenEyes => 0.064f;

            public Vector3DFloat ImageCenter => new Vector3DFloat() { X = 0.0f, Y = 0.0f, Z = 0.25f };

            public float ImageWidth => 0.1f;

            public float ImageHeight => 0.1f;

            public int EyeImageWidthPixels = 32;

            public int EyeImageHeightPixels = 32;

            public int DependantDetectorsRangeWidthCount = 50;

            public int DependantDetectorsRangeHeightCount = 50;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public float MemoryClustersThreshold => 0.66f;
        }        
    }
}


//private (float centerX, float centerY) GetPointOnMnistImage(Vector3DFloat pupil, Direction eyeDirection, Direction imageNormalDirection)
//{
//    // Входные данные
//    // Координаты точки A и углы линии Л
//    float Ax = pupil.X, Ay = pupil.Y, Az = pupil.Z;
//    float lineAngleXZ = eyeDirection.XRadians; // угол в плоскости XZ
//    float lineAngleYZ = eyeDirection.YRadians; // угол в плоскости YZ

//    // Координаты точки B и углы нормали плоскости П
//    float Bx = Constants.ImageCenter.X, By = Constants.ImageCenter.Y, Bz = Constants.ImageCenter.Z;
//    float normalAngleXZ = imageNormalDirection.XRadians; // угол в плоскости XZ
//    float normalAngleYZ = imageNormalDirection.YRadians; // угол в плоскости YZ

//    // Направляющий вектор линии Л
//    float lineDirX = MathF.Sin(lineAngleXZ);
//    float lineDirY = MathF.Sin(lineAngleYZ);
//    float lineDirZ = MathF.Cos(lineAngleXZ) * MathF.Cos(lineAngleYZ);

//    // Направляющий вектор нормали плоскости П
//    float normalX = MathF.Sin(normalAngleXZ);
//    float normalY = MathF.Sin(normalAngleYZ);
//    float normalZ = MathF.Cos(normalAngleXZ) * MathF.Cos(normalAngleYZ);

//    // Вектор точки A -> точки B
//    float ABx = Bx - Ax;
//    float ABy = By - Ay;
//    float ABz = Bz - Az;

//    // Скалярное произведение направляющего вектора линии и нормали
//    float dotProduct = lineDirX * normalX + lineDirY * normalY + lineDirZ * normalZ;

//    // Проверка на параллельность линии и плоскости
//    if (MathF.Abs(dotProduct) < 1e-6)
//    {
//        Console.WriteLine("Линия и плоскость параллельны или лежат в одной плоскости.");
//        return (float.NaN, float.NaN);
//    }

//    // Параметр t для точки пересечения
//    float t = (ABx * normalX + ABy * normalY + ABz * normalZ) / dotProduct;

//    // Координаты точки пересечения в трехмерном пространстве
//    float intersectX = Ax + t * lineDirX;
//    float intersectY = Ay + t * lineDirY;
//    float intersectZ = Az + t * lineDirZ;

//    // Вектор из точки B до точки пересечения
//    float BIntersectX = intersectX - Bx;
//    float BIntersectY = intersectY - By;
//    float BIntersectZ = intersectZ - Bz;

//    // Преобразование в двумерные координаты на плоскости П
//    // Ось X плоскости
//    float planeXDirX = MathF.Cos(normalAngleXZ);
//    float planeXDirY = 0.0f;
//    float planeXDirZ = -MathF.Sin(normalAngleXZ);

//    // Ось Y плоскости
//    (float planeYDirX, float planeYDirY, float planeYDirZ) = MathHelper.VectorProduct(normalX, normalY, normalZ, planeXDirX, planeXDirY, planeXDirZ);

//    //float l = MathF.Sqrt(planeYDirX * planeYDirX + planeYDirY * planeYDirY + planeYDirZ * planeYDirZ);
//    //planeYDirX /= l;
//    //planeYDirY /= l;
//    //planeYDirZ /= l;

//    // Координаты в системе плоскости
//    float planeCoordX = BIntersectX * planeXDirX + BIntersectY * planeXDirY + BIntersectZ * planeXDirZ;
//    float planeCoordY = BIntersectX * planeYDirX + BIntersectY * planeYDirY + BIntersectZ * planeYDirZ;

//    planeCoordX -= Constants.ImageCenter.X - Constants.ImageWidth / 2;
//    planeCoordY -= Constants.ImageCenter.Y - Constants.ImageHeight / 2;

//    planeCoordX /= Constants.ImageWidth / MNISTHelper.MNISTImageWidthPixels;
//    planeCoordY /= Constants.ImageHeight / MNISTHelper.MNISTImageHeightPixels;

//    return (planeCoordX, planeCoordY);
//}


