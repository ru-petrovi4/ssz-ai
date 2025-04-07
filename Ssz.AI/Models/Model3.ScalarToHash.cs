using Avalonia.Controls;
using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using Newtonsoft.Json;
using OpenCvSharp;
using OxyPlot;
using OxyPlot.Avalonia;
using OxyPlot.Axes;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.AI.Views;
using System;
using System.Collections;
using System.Collections.Generic;
using System.DrawingCore;
using System.Linq;
using System.Numerics.Tensors;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model3
    {   
        /// <summary>
        ///     Скаляр в хэш
        /// </summary>
        public Model3(Random random)
        {
            BaseHash = new float[HashLength];
            CompareHash = new float[HashLength];
            LongCodeProjectionToHash = new int[LongCodeLength];

            foreach (int i in Enumerable.Range(0, LongCodeProjectionToHash.Length))
            {
                LongCodeProjectionToHash[i] = random.Next(HashLength);
            }
        }        

        public const int HashLength = 200;

        public const int LongCodeLength = 1000;

        /// <summary>
        ///     Расстояние между исполинами
        /// </summary>
        public float K0 { get; set; }

        /// <summary>
        ///     Диапазон ближайших исполинов
        /// </summary>
        public float K1 { get; set; }

        /// <summary>
        ///     Диапазон ближаших
        /// </summary>
        public float K2 { get; set; }

        /// <summary>
        ///     Расстояние между исполинами
        /// </summary>
        public float K3 { get; set; }

        public VisualizationWithDesc[] GetImageWithDescs0(float value)
        {
            return [
                new Plot2DWithDesc { Model = CreatePlotModel(value),
                    Desc = $"Косинусное расстояние до других значений" },                
                ];
        }

        private PlotModel CreatePlotModel(float value)
        {
            var model = new PlotModel { Title = "Косинусное расстояние" };

            model.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = AxisPosition.Left,
                Title = "Косинусное расстояние",
                Minimum = 0,
                Maximum = 1
            });

            model.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = AxisPosition.Bottom,
                Title = "Величина",
                Minimum = 0,
                Maximum = 1
            });

            var series = new OxyPlot.Series.LineSeries { MarkerType = MarkerType.Circle };            

            ComputeHash(value, BaseHash);
            float delta = 1.0f / LongCodeProjectionToHash.Length;
            for (float v = 0.0f; v < 1.0; v += delta)
            {
                ComputeHash(v, CompareHash);
                float cosineSimilarity = TensorPrimitives.CosineSimilarity(BaseHash, CompareHash);
                if (float.IsNaN(cosineSimilarity) || float.IsInfinity(cosineSimilarity))
                    cosineSimilarity = 0.0f;
                series.Points.Add(new DataPoint(v, cosineSimilarity));
            }

            model.Series.Add(series);
            return model;
        }

        private void ComputeHash(float value, float[] hash)
        {
            Array.Clear(hash);

            float delta = K0;
            for (float v = value - K1; v < value + K1; v += delta)
            {
                if (v >= 0.0f && v < 1.0f)
                {
                    int i = (int)(v / delta);
                    if (i < LongCodeProjectionToHash.Length)
                        hash[LongCodeProjectionToHash[i]] = 1.0f;
                }
            }

            delta = 1.0f / LongCodeProjectionToHash.Length;
            for (float v = value - K2; v < value + K2; v += delta)
            {
                if (v >= 0.0f && v < 1.0f)
                {
                    int i = (int)(v / delta);
                    if (i < LongCodeProjectionToHash.Length)
                        hash[LongCodeProjectionToHash[i]] = 1.0f;
                }
            }
        }

        private float[] BaseHash = null!;
        private float[] CompareHash = null!;
        private int[] LongCodeProjectionToHash = null!;
    }
}
