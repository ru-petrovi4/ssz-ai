using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Logging;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel;

namespace Ssz.AI.Views.AdvancedEmbeddingViews;

public partial class Model03View : UserControl
{
    public Model03View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        Model = new Model03();
        Task.Run(() =>
        {
            Model.Find_ClustersOneToOneMatcher_MappingLinear();
            //Model.Find_ClustersOneToOneMatcher_MappingLinear();            
        });
    }

    public Model03 Model = null!;
}