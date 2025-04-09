using Avalonia.Controls;
using Avalonia.Layout;
using Ssz.AI.Views;

namespace Ssz.AI.Helpers
{
    public static class VisualisationHelper
    {
        public static void ShowImages(System.Drawing.Image[] images)
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

            for (int i = 0; i < images.Length && i < 10; i += 1)
            {
                var bitmap = BitmapHelper.ConvertImageToAvaloniaBitmap(images[i]);
                var imageControl = new Avalonia.Controls.Image
                {
                    Source = bitmap,
                    //Width = 150,
                    //Height = 150
                };
                panel.Children.Add(imageControl);
            }

            window.Show();
        }
    }
}


//public static void ShowImages(List<Mat> images)
//{
//    var window = new MainWindow
//    {
//        Width = 1500,
//        Height = 600,
//        Content = new StackPanel
//        {
//            Orientation = Orientation.Horizontal
//        }
//    };

//    var panel = (StackPanel)window.Content;

//    for (int i = 0; i < images.Count && i < 10; i += 1)
//    {
//        var image = images[i];
//        var bitmap = BitmapHelper.ConvertMatToAvaloniaBitmap(image);
//        var imageControl = new Avalonia.Controls.Image
//        {
//            Source = bitmap,
//            Width = 150,
//            Height = 150
//        };
//        panel.Children.Add(imageControl);
//    }

//    window.Show();
//}
