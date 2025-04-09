using OxyPlot;
using Ssz.AI.Models;
using Ssz.Utils.Avalonia.Model3D;
using System.Drawing;

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

    public class Model3DWithDesc : VisualizationWithDesc
    {
        public Model3DScene Data { get; set; } = null!;
    }

    public class Plot2DWithDesc : VisualizationWithDesc
    {
        public PlotModel Model { get; set; } = null!;
    }
}
