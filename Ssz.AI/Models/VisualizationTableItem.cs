using System.Collections.Generic;
using System.Drawing;

namespace Ssz.AI.Models
{
    public class VisualizationTableItem
    {
        public float[] Hash { get; init; } = null!;

        public Color Color { get; set; }

        public Bitmap Image { get; set; } = null!;

        public float[][] SubArea_MiniColumns_Hashes { get; init; } = null!;
    }    
}
