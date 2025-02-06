using System.DrawingCore;

namespace Ssz.AI.ViewModels
{
    public class ImageWithDesc
    {
        public Avalonia.Media.Imaging.Bitmap Image { get; set;} = null!;

        public string Desc { get; set; } = @"";
    }
}
