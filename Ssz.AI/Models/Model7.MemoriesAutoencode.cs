﻿ using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model7
    {
        #region construction and destruction

        /// <summary>
        ///     Гистограммы для миниколонок
        /// </summary>
        public Model7()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            (Labels, Images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            GradientMatricesCollection = new(Images.Length);
            foreach (int i in Enumerable.Range(0, Images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(Images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                GradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            Retina = new Retina(Constants, gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);

            Cortex = new Cortex(Constants, Retina);            
            
            CurrentMnistImageIndex = -1; // Перед первым элементом

            // Прогон картинок
            DoSteps_MNIST(5000);

            FindAutoencoder(Cortex.MiniColumns[Cortex.MiniColumns.GetLength(0) / 2, Cortex.MiniColumns.GetLength(1) / 2]);
        }        

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();

        public readonly byte[] Labels;
        public readonly byte[][] Images;
        public readonly List<GradientInPoint[,]> GradientMatricesCollection;
        public int CurrentMnistImageIndex = 0;

        public readonly Retina Retina;

        public readonly Cortex Cortex;        

        public int Generated_CenterX { get; set; }
        public int Generated_CenterXDelta { get; set; }
        public int Generated_CenterY { get; set; }
        public double Generated_AngleDelta { get; set; }
        public double Generated_Angle { get; set; }        

        public Image[] GetImages1(double positionK, double angleK)
        {   
            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            ActivitiyMaxInfo activitiyMaxInfo = new();
                
            //GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.GetLength(0) * Retina.Detectors.GetLength(1));
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.GetLength(0)))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
                Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
                Cortex.MiniColumns.GetLength(0) / 2,
                Cortex.MiniColumns.GetLength(1) / 2,
                Cortex.SubAreaMiniColumnsRadius + 2);
            //var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }

        private (GradientInPoint[,], Bitmap) GetGeneratedLine_gradientMatrix(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            Generated_CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0);
            Generated_CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + Generated_CenterXDelta;
            Generated_CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            Generated_AngleDelta = angleK * 2.0 * Math.PI;
            Generated_Angle = Math.PI / 2 + Generated_AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(Generated_CenterX + lineLength * Math.Cos(Generated_Angle));
            int endY = (int)(Generated_CenterY + lineLength * Math.Sin(Generated_Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(Generated_CenterX - lineLength * Math.Cos(Generated_Angle));
            int startY = (int)(Generated_CenterY - lineLength * Math.Sin(Generated_Angle));

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
            return (SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), resizedBitmap);
        }

        public Image[] GetImages2()
        {
            //var totalMnistBitmap = GetMnistTotalBitmap();

            var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            ActivitiyMaxInfo activitiyMaxInfo = new();

            //GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.GetLength(0) * Retina.Detectors.GetLength(1));
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.GetLength(0)))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
                Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
                Cortex.MiniColumns.GetLength(0) / 2,
                Cortex.MiniColumns.GetLength(1) / 2,
                Cortex.SubAreaMiniColumnsRadius + 2);
            //var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            var originalBitmap = MNISTHelper.GetBitmap(Images[CurrentMnistImageIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            return [originalBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }        

        public Image[] GetImages3()
        {
            var image = Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex);

            return [ image ];
        }

        public void DoSteps_MNIST(int stepsCount)
        {
            var random = new Random();

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

            dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2 = new ulong[Constants.CortexWidth, Constants.CortexHeight, Constants.HashLength];

            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentMnistImageIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

                DoStep(gradientMatrix, dataToDisplayHolder, random);
            }
        }

        public void DoStep_GeneratedLine(double positionK, double angleK)
        {
            var random = new Random();

            ActivitiyMaxInfo activitiyMaxInfo = new();

            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);
            
            //DoStep(gradientMatrix, activitiyMaxInfo, random);            
        }

        #endregion

        private void DoStep(GradientInPoint[,] gradientMatrix, DataToDisplayHolder dataToDisplayHolder, Random random)
        {
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    });

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.CalculateHash(mc.Temp_Hash);

                    int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);                    
                    dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2[mc.MCX, mc.MCY, bitsCountInHash] += 1;

                    if (bitsCountInHash >= 11)
                    {
                        mc.Memories.Add(new Memory { Hash = (float[])mc.Temp_Hash.Clone() });
                    }
                });
        }

        private void FindAutoencoder(MiniColumn miniColumn)
        {            
            var autoencoder = new Autoencoder(inputSize: Constants.HashLength, bottleneckSize: 50, maxActiveUnits: 7);

            float prevBinaryCrossEntropy = float.MaxValue;
            float binaryCrossEntropy;
            float binaryCrossEntropyDelta = 1.0f;

            int trainCount = (int)(miniColumn.Memories.Count * 0.9);

            int iterationsCount = 0;
            int stopIterationsCount = 0;
            while (stopIterationsCount < 5)
            {
                binaryCrossEntropy = 0.0f;

                foreach (var memory in miniColumn.Memories.Take(trainCount))
                {
                    var input = new DenseTensor<float>(memory.Hash);

                    binaryCrossEntropy += autoencoder.Train(input, learningRate: 0.01f);
                }

                binaryCrossEntropy = binaryCrossEntropy / miniColumn.Memories.Count;

                binaryCrossEntropyDelta = prevBinaryCrossEntropy - binaryCrossEntropy;
                if (binaryCrossEntropyDelta > -0.00001f && binaryCrossEntropyDelta < 0.00001f)
                    stopIterationsCount += 1;
                else
                    stopIterationsCount = 0;

                iterationsCount += 1;

                prevBinaryCrossEntropy = binaryCrossEntropy;                
            }

            binaryCrossEntropy = 0.0f;

            int memoriesCount = 0;
            foreach (var memory in miniColumn.Memories.Skip(trainCount))
            {
                var input = new DenseTensor<float>(memory.Hash);

                binaryCrossEntropy += autoencoder.ComputeBinaryCrossEntropy(input);

                float sum = TensorPrimitives.Sum(autoencoder.EncoderOutput.Buffer);

                memoriesCount += 1;
            }

            if (memoriesCount > 0)
                binaryCrossEntropy = binaryCrossEntropy / memoriesCount;
        }

        public static readonly Color[] DefaultColors =
        {
            Color.FromArgb(0xFF, 0x00, 0xFE),
            Color.FromArgb(0x02, 0x00, 0xF9),
            Color.FromArgb(0x00, 0xFF, 0xFF),
            Color.FromArgb(0xFF, 0x80, 0x41),
            Color.FromArgb(0xFC, 0x01, 0x00),
            Color.FromArgb(0x00, 0xFF, 0x01),
            Color.FromArgb(0xFF, 0xFF, 0x00),
            Color.FromArgb(0xFF, 0x00, 0x00),            
        };

        private class VisualizationTableItemsCluster
        {
            public float[] Hash = null!;

            public float[] TempHash = null!;

            public List<VisualizationTableItem> VisualizationTableItems = null!;

            public float MinCosineSimilarity;

            public float MaxCosineSimilarity;            
        }

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {
            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidth => MNISTHelper.MNISTImageWidth;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeight => MNISTHelper.MNISTImageHeight;

            public int AngleRangesCount => 6;

            public int MagnitudeRangesCount => 4;

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
            ///     Расстояние между детекторами по коризонтали и вертикали  
            /// </summary>
            public double DetectorDelta => 0.1;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;            

            public int HashLength => 200;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => null;

            /// <summary>
            ///     Индекс X центра подобласти [0..CortexWidth]
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти [0..CortexHeight]
            /// </summary>
            public int SubAreaCenter_Cy => 100;           

            /// <summary>
            ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
            /// </summary>
            public int InitialMemoryBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 8;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int NearestMiniColumnsDelta => 7;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;
        }        
    }
}


//private void FindAutoencoder(MiniColumn miniColumn)
//{
//    // Параметры модели
//    int inputSize = 200;
//    int hiddenSize = 50;
//    int batchSize = 32;
//    int epochs = 20;

//    // Создание весов и смещений
//    var weights_encoder = tf.Variable(tf.random.normal((inputSize, hiddenSize)), name: "weights_encoder", trainable: true);
//    var biases_encoder = tf.Variable(tf.zeros(hiddenSize), name: "biases_encoder", trainable: true);

//    var weights_decoder = tf.Variable(tf.random.normal((hiddenSize, inputSize)), name: "weights_decoder", trainable: true);
//    var biases_decoder = tf.Variable(tf.zeros(inputSize), name: "biases_decoder", trainable: true);

//    if (weights_encoder == null || biases_encoder == null || weights_decoder == null || biases_decoder == null)
//    {
//        Console.WriteLine("Ошибка: один из параметров модели не инициализирован.");
//        return;
//    }

//    // Функция прямого прохода
//    Func<NDArray, NDArray> forward = input =>
//    {
//        //var encoded = tf.sigmoid(tf.matmul(input, weights_encoder) + biases_encoder);
//        //var decoded = tf.sigmoid(tf.matmul(encoded, weights_decoder) + biases_decoder);
//        //return decoded.numpy();

//        var inputTensor = tf.convert_to_tensor(input);
//        var encoded = tf.sigmoid(tf.matmul(inputTensor, weights_encoder) + biases_encoder);
//        var decoded = tf.sigmoid(tf.matmul(encoded, weights_decoder) + biases_decoder);
//        return decoded.numpy();
//    };

//    // Функция потерь
//    Func<NDArray, NDArray, Tensor> loss_fn = (inputs, outputs) =>
//    {
//        //return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: inputs, logits: outputs));

//        var inputsTensor = tf.convert_to_tensor(inputs);
//        var outputsTensor = tf.convert_to_tensor(outputs);
//        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels: inputsTensor, logits: outputsTensor));
//    };

//    // Оптимизатор
//    var optimizer = tf.keras.optimizers.Adam(learning_rate: 0.001f);

//    // Генерация данных для тренировки
//    var trainData = GenerateBooleanData(10000, inputSize, 15);

//    // Обучение модели
//    for (int epoch = 0; epoch < epochs; epoch++)
//    {
//        foreach (var batch in GetBatches(trainData, batchSize))
//        {
//            using (var tape = tf.GradientTape())
//            {
//                tape.watch(weights_encoder);
//                tape.watch(biases_encoder);
//                tape.watch(weights_decoder);
//                tape.watch(biases_decoder);

//                var outputs = forward(batch);
//                var loss = loss_fn(batch, outputs);

//                if (loss == null)
//                {
//                    Console.WriteLine("Ошибка: Loss не был рассчитан.");
//                    return;
//                }

//                tape.watch(loss);

//                // Вычисление и применение градиентов
//                var gradients = tape.gradient(loss, new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder });
//                if (gradients == null || gradients.Contains(null))
//                {
//                    Console.WriteLine("Ошибка: градиенты не были рассчитаны.");
//                    return;
//                }

//                var zipped = gradients.Zip(new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder });
//                optimizer.apply_gradients(zipped); // zip(gradients, new[] { weights_encoder, biases_encoder, weights_decoder, biases_decoder })
//            }
//        }

//        // Вывод текущей ошибки
//        var epochLoss = loss_fn(trainData, forward(trainData));
//        Console.WriteLine($"Эпоха {epoch + 1}/{epochs}, Потеря: {epochLoss.numpy()}");
//    }
//}

//static NDArray GenerateBooleanData(int samples, int size, int onesPerVector)
//{
//    Random rnd = new Random();
//    var data = np.zeros((samples, size));

//    for (int i = 0; i < samples; i++)
//    {
//        var indices = new int[size];
//        for (int j = 0; j < size; j++)
//            indices[j] = j;

//        for (int j = 0; j < onesPerVector; j++)
//        {
//            int index = rnd.Next(size - j);
//            data[i, indices[index]] = 1.0;
//            indices[index] = indices[size - j - 1];
//        }
//    }

//    return data.astype(np.float32);
//}

//static IEnumerable<NDArray> GetBatches(NDArray data, int batchSize)
//{
//    for (int i = 0; i < data.shape[0]; i += batchSize)
//    {
//        yield return data[$"{i}:{Math.Min(i + batchSize, data.shape[0])}"];
//    }
//}