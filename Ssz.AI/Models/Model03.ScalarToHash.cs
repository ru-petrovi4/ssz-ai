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
using System.Drawing;
using System.Linq;
using System.Numerics.Tensors;
using Size = System.Drawing.Size;

namespace Ssz.AI.Models
{
    public class Model3
    {   
        /// <summary>
        ///     Скаляр в хэш
        /// </summary>
        public Model3(Random random)
        {
            _baseHash = new float[HashLength];
            _compareHash = new float[HashLength];
            _longCode_ToHashIndices = new int[LongCodeLength];

            foreach (int i in Enumerable.Range(0, _longCode_ToHashIndices.Length))
            {
                _longCode_ToHashIndices[i] = random.Next(HashLength);
            }
        }        

        public const int HashLength = 300;

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
        ///     
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

            if (K0 > 0.0f)
            {                
                int bigsCount = (int)(1.0f / K0);
                var value_BigToHashIndices = new int[bigsCount];
                foreach(int i in Enumerable.Range(0, bigsCount))
                {
                    value_BigToHashIndices[i] = _longCode_ToHashIndices[(int)(K0 * i * _longCode_ToHashIndices.Length)];
                }

                Array.Clear(_baseHash);
                Hash.ValueToHash(
                    value,
                    value_BigToHashIndices,
                    _longCode_ToHashIndices,
                    bigRadius: K1,
                    smallRadius: K2,                    
                    _baseHash);
                float delta = 1.0f / _longCode_ToHashIndices.Length;
                for (float v = 0.0f; v < 1.0; v += delta)
                {
                    Array.Clear(_compareHash);
                    Hash.ValueToHash(
                        v,
                        value_BigToHashIndices,
                        _longCode_ToHashIndices,
                        bigRadius: K1,
                        smallRadius: K2,                        
                        _compareHash);
                    float cosineSimilarity = TensorPrimitives.CosineSimilarity(_baseHash, _compareHash);
                    if (float.IsNaN(cosineSimilarity) || float.IsInfinity(cosineSimilarity))
                        cosineSimilarity = 0.0f;
                    series.Points.Add(new DataPoint(v, cosineSimilarity));
                }
            }            

            model.Series.Add(series);
            return model;
        }

        private float[] _baseHash = null!;
        private float[] _compareHash = null!;
        private int[] _longCode_ToHashIndices = null!;
    }
}
