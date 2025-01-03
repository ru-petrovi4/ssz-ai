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
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using static Ssz.AI.Models.Cortex;
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
            Logger = ActivatorUtilities.CreateInstance<Logger<Model7>>(Program.Host.Services);

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
            CollectMemories_MNIST(5000);

            Task.Factory.StartNew(() =>
            {
                //TestSerialization();
                CalculateAutoencoders();
            }, TaskCreationOptions.LongRunning);

                //FindHyperColumn();            
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

        public CancellationTokenSource Temp_StopAutoencoderFinding_CancellationTokenSource  { get; set; } = new();

        public ILogger Logger { get; }

        public Image[] GetImages1(double positionK, double angleK)
        {
            //(GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);

            //var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            //ActivitiyMaxInfo activitiyMaxInfo = new();

            ////GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            //List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            //foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
            //    foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
            //    {
            //        Detector d = Retina.Detectors[dx, dy];
            //        if (d.Temp_IsActivated)
            //            activatedDetectors.Add(d);
            //    }
            //var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
            //    Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
            //    Cortex.MiniColumns.Dimensions[0] / 2,
            //    Cortex.MiniColumns.Dimensions[1] / 2,
            //    Cortex.SubAreaMiniColumnsRadius + 2);            

            //return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
            return [];
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
            ////var totalMnistBitmap = GetMnistTotalBitmap();

            //var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

            //var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            //ActivitiyMaxInfo activitiyMaxInfo = new();

            ////GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            //List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            //foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
            //    foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
            //    {
            //        Detector d = Retina.Detectors[dx, dy];
            //        if (d.Temp_IsActivated)
            //            activatedDetectors.Add(d);
            //    }
            //var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
            //    Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
            //    Cortex.MiniColumns.Dimensions[0] / 2,
            //    Cortex.MiniColumns.Dimensions[1] / 2,
            //    Cortex.SubAreaMiniColumnsRadius + 2);
            ////var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            //var originalBitmap = MNISTHelper.GetBitmap(Images[CurrentMnistImageIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            //return [originalBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
            return [];
        }        

        public Image[] GetImages3()
        {
            var image = Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex);

            return [ image ];
        }

        public void CollectMemories_MNIST(int stepsCount)
        {
            var random = new Random();

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

            dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2 = new ulong[Constants.CortexWidth, Constants.CortexHeight, Constants.HashLength];

            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentMnistImageIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

                DoStep_CollectMemories_MNIST(gradientMatrix, dataToDisplayHolder, random);
            }
        }

        public void DoStep_GeneratedLine(double positionK, double angleK)
        {
            var random = new Random();            

            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);
            
            //DoStep(gradientMatrix, activitiyMaxInfo, random);            
        }        

        #endregion

        private void CalculateAutoencoders()
        {
            var cancellationToken =  Temp_StopAutoencoderFinding_CancellationTokenSource.Token;

            const string fileName = @"Data\cortex.bin";

            if (File.Exists(fileName))
            {
                using (var stream = new FileStream(fileName, FileMode.Open))
                using (var reader = new SerializationReader(stream))
                {
                    reader.ReadOwnedDataSerializable(Cortex, "autoencoders");
                }
            }            

            var miniColumnsToProcess = Cortex.MiniColumns.Data.Where(mc => mc is not null && mc.Autoencoder is null).ToArray();

            Logger.LogInformation($"CalculateAutoencoders(...) started; Count: {miniColumnsToProcess.Length}");

            Stopwatch sw = Stopwatch.StartNew();

            Parallel.For(
                fromInclusive: 0,
                toExclusive: miniColumnsToProcess.Length,
                (mci, s) =>
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        s.Stop();
                        return;
                    }
                        
                    MiniColumn miniColumn = miniColumnsToProcess[mci];
                    miniColumn.Autoencoder = FindAutoencoder(miniColumn);

                    Logger.LogInformation($"FindAutoencoder(...) finished; {mci + 1}/{miniColumnsToProcess.Length}; TrainingDurationMilliseconds: {miniColumn.Autoencoder.TrainingDurationMilliseconds}; ControlCosineSimilarity: {miniColumn.Autoencoder.ControlCosineSimilarity}");
                });

            Logger.LogInformation($"CalculateAutoencoders(...) ElapsedMilliseconds: {sw.ElapsedMilliseconds}; Count: {miniColumnsToProcess.Length}");

            using (var memoryStream = new MemoryStream(1024 * 1024))
            {
                var isEmpty = false;
                using (var writer = new SerializationWriter(memoryStream, true))
                {
                    writer.WriteOwnedDataSerializable(Cortex, "autoencoders");                    
                }

                if (!isEmpty)
                    using (FileStream fileStream = File.Create(fileName))
                    {
                        memoryStream.WriteTo(fileStream);
                    }
            }
        }

        private void TestSerialization()
        {
            const string fileName = @"Data\cortex_test.bin";

            Autoencoder autoencoder = new(inputSize: 200, bottleneckSize: 50, maxActiveUnits: 11);           

            using (var memoryStream = new MemoryStream(1024 * 1024))
            {
                var isEmpty = false;
                using (var writer = new SerializationWriter(memoryStream, true))
                {
                    writer.WriteOwnedDataSerializableAndRecreatable(autoencoder, null);
                }

                if (!isEmpty)
                    using (FileStream fileStream = File.Create(fileName))
                    {
                        memoryStream.WriteTo(fileStream);
                    }
            }

            Autoencoder? deserializedAutoencoder;

            if (File.Exists(fileName))
            {
                using (var stream = new FileStream(fileName, FileMode.Open))
                using (var reader = new SerializationReader(stream))
                {
                    deserializedAutoencoder = reader.ReadOwnedDataSerializableAndRecreatable<Autoencoder>(null);
                }
            }
        }

        private void FindHyperColumn()
        {
            //MiniColumn winnerMiniColumn = Cortex.MiniColumns[0, 0];
            //foreach (int mcy in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[1]))
            //    foreach (int mcx in Enumerable.Range(0, Cortex.MiniColumns.Dimensions[0]))
            //    {
            //        var mc = Cortex.MiniColumns[mcx, mcy];
            //        if (mc.Memories.Count > winnerMiniColumn.Memories.Count)
            //            winnerMiniColumn = mc;
            //    }

            //foreach (int mcx in Enumerable.Range(winnerMiniColumn.MCX, Cortex.MiniColumns.Dimensions[0] - winnerMiniColumn.MCX))
            //{
            //    var mc = Cortex.MiniColumns[mcx, winnerMiniColumn.MCY];
            //    if (mc.Memories.Count > winnerMiniColumn.Memories.Count)
            //        winnerMiniColumn = mc;
            //}

            FindAutoencoder(Cortex.MiniColumns[Cortex.MiniColumns.Dimensions[0] / 3, Cortex.MiniColumns.Dimensions[1] / 3]);
        }

        private void DoStep_CollectMemories_MNIST(GradientInPoint[,] gradientMatrix, DataToDisplayHolder dataToDisplayHolder, Random random)
        {
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Retina.Detectors.Data.Length,
                    di =>
                    {
                        var d = Retina.Detectors.Data[di];
                        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    });

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.MiniColumns.Data.Length,
                mci =>
                {
                    var mc = Cortex.MiniColumns.Data[mci];
                    if (mc is not null)
                    {
                        mc.CalculateHash(mc.Temp_Hash);

                        int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);
                        dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2[mc.MCX, mc.MCY, bitsCountInHash] += 1;

                        if (bitsCountInHash >= 11)
                        {
                            mc.Memories.Add(new Memory { Hash = (float[])mc.Temp_Hash.Clone() });
                        }
                    }                    
                });
        }        

        private Autoencoder FindAutoencoder(MiniColumn miniColumn)
        {            
            var autoencoder = new Autoencoder(inputSize: Constants.HashLength, bottleneckSize: 50, maxActiveUnits: 11);

            Stopwatch sw = Stopwatch.StartNew();

            autoencoder.CosineSimilarity = float.MaxValue;
            float cosineSimilarity = 1.0f;
            float cosineSimilarityDelta = 1.0f;

            int trainCount = (int)(miniColumn.Memories.Count * 0.9);
            int memoriesCount = 0;

            autoencoder.IterationsCount = 0;
            int stopIterationsCount = 0;
            while (stopIterationsCount < 5)
            {
                cosineSimilarity = 0.0f;
                
                memoriesCount = 0;
                foreach (var memory in miniColumn.Memories.Take(trainCount))
                {
                    var input = new DenseTensor<float>(memory.Hash);

                    float cs = autoencoder.Train(input, learningRate: 0.01f);
                    if (!float.IsNaN(cs))
                    {
                        cosineSimilarity += cs;
                        memoriesCount += 1;
                    }
                    else
                    {
                    }
                }

                if (memoriesCount > 0)
                    cosineSimilarity = cosineSimilarity / memoriesCount;

                cosineSimilarityDelta = cosineSimilarity - autoencoder.CosineSimilarity;
                if (cosineSimilarityDelta > -0.0001f && cosineSimilarityDelta < 0.0001f)
                    stopIterationsCount += 1;
                else
                    stopIterationsCount = 0;

                autoencoder.IterationsCount += 1;

                autoencoder.CosineSimilarity = cosineSimilarity;                
            }

            cosineSimilarity = 0.0f;

            memoriesCount = 0;
            foreach (var memory in miniColumn.Memories.Skip(trainCount))
            {
                var input = new DenseTensor<float>(memory.Hash);

                float cs = autoencoder.ComputeCosineSimilarity(input);
                if (!float.IsNaN(cs))
                {
                    cosineSimilarity += cs;
                    memoriesCount += 1;
                }
                else
                {
                }
                //float sum = TensorPrimitives.Sum(autoencoder.Bottleneck.Buffer);
            }

            if (memoriesCount > 0)
                autoencoder.ControlCosineSimilarity = cosineSimilarity / memoriesCount;
            else
                autoencoder.ControlCosineSimilarity = 0;

            sw.Stop();
            autoencoder.TrainingDurationMilliseconds = sw.ElapsedMilliseconds;            

            return autoencoder;
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
            ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
            /// </summary>
            public int InitialMemoryBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 8;           

            /// <summary>
            ///     Примерное количество воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 1;

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
            public int NearestMiniColumnsDelta => 1;
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