using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using OpenCvSharp;
using OxyPlot;
using OxyPlot.Series;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.IO;

namespace Ssz.AI.Models
{
    public class DetectorVisualization
    {
        private List<Detector> detectors;

        public DetectorVisualization(List<Detector> detectors)
        {
            this.detectors = detectors;
        }

        public void Visualize(List<Mat> images)
        {
            var scalarProducts = new List<double>();
            List<int>? firstImageActivations = null;

            for (int i = 0; i < images.Count; i++)
            {
                var image = images[i];
                var imageActivations = detectors.ConvertAll(d => d.IsActivated(image) ? 1 : 0);

                if (firstImageActivations != null)
                {
                    var scalarProduct = 0.0;
                    for (int j = 0; j < imageActivations.Count; j++)
                    {
                        scalarProduct += imageActivations[j] * firstImageActivations[j];
                    }

                    scalarProducts.Add(scalarProduct);
                    //Console.WriteLine($"{Math.Round(i * 0.1, 1)}, SP = {Math.Round(scalarProduct, 1)}");
                }
                else
                {
                    firstImageActivations = imageActivations;
                }
            }

            ShowGraph(scalarProducts);
            ShowImages(images, scalarProducts);
        }

        private void ShowGraph(List<double> scalarProducts)
        {
            var plotModel = new PlotModel { Title = "Scalar Products" };
            var series = new LineSeries();

            for (int i = 0; i < scalarProducts.Count; i++)
            {
                series.Points.Add(new DataPoint(i * 0.1, scalarProducts[i]));
            }

            plotModel.Series.Add(series);

            var window = new MainWindow
            {
                Width = 600,
                Height = 400,
                Content = new OxyPlot.Avalonia.PlotView
                {
                    Model = plotModel
                }
            };
            window.Show();
        }

        private void ShowImages(List<Mat> images, List<double> scalarProducts)
        {
            var window = new MainWindow
            {
                Width = 1500,
                Height = 600,
                Content = new StackPanel
                {
                    Orientation = Orientation.Horizontal
                }
            };

            var panel = (StackPanel)window.Content;

            for (int i = 0; i < images.Count && i < 10; i += 2)
            {
                var image = images[i];
                var bitmap = BitmapHelper.ConvertMatToBitmap(image);
                var imageControl = new Image
                {
                    Source = bitmap,
                    Width = 150,
                    Height = 150
                };

                panel.Children.Add(imageControl);
            }

            window.Show();
        }
    }
}
