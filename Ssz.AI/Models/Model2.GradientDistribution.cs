using Avalonia.Controls;
using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.Linq;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model2
    {
        /// <summary>
        ///     Построение графика распределения величин градиентов
        /// </summary>
        public Model2()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();            

            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobelObsoslete(images[i], MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
                //SobelOperator.CalculateDistributionObsolete(gradientMatrix, constants, gradientDistribution);
            }

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            dataToDisplayHolder.GradientDistribution = gradientDistribution;
        }
    }
}
