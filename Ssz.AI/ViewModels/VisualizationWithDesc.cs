using Ssz.AI.Models;
using System.DrawingCore;

namespace Ssz.AI.ViewModels
{
    public class VisualizationWithDesc
    {
        public string Desc { get; set; } = @"";
    }

    public class ImageWithDesc : VisualizationWithDesc
    {
        public Avalonia.Media.Imaging.Bitmap Image { get; set;} = null!;
    }

    public class Model3DViewWithDesc : VisualizationWithDesc
    {
        public Model3DScene Data { get; set; } = null!;
    }
}
