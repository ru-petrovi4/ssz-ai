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

public partial class Model01View : UserControl
{
    public Model01View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        Model = new Model01();
        Task.Run(() =>
        {            
            Model.FindDiscreteEmbeddings_LanguageDiscreteEmbeddings_Object();
            //Model.GetEmbeddingsQualityInfo2();
        });
    }

    public Model01 Model = null!;
}