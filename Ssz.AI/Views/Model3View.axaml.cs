using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Views;

public partial class Model3View : UserControl
{
    public Model3View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        _random = new Random(1);

        _model = new Model3(_random);
        
        LevelScrollBar0.ValueChanged += (s, e) => OnLevelScrollBarChanged();
        LevelScrollBar1.ValueChanged += (s, e) => OnLevelScrollBarChanged();
        LevelScrollBar2.ValueChanged += (s, e) => OnLevelScrollBarChanged();
        //LevelScrollBar3.ValueChanged += (s, e) => OnLevelScrollBarChanged();
        OnLevelScrollBarChanged();        
    } 

    private void OnLevelScrollBarChanged()
    {        
        _model.K0 = (float)LevelScrollBar0.Value;
        _model.K1 = (float)LevelScrollBar1.Value;
        _model.K2 = (float)LevelScrollBar2.Value;
        //_model.K3 = (float)LevelScrollBar3.Value;
        Refresh_ImagesSet0();
    }

    private void ValueScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh_ImagesSet0();
    }

    private void Refresh_ImagesSet0()
    {        
        ImagesSet0.MainItemsControl.ItemsSource = _model.GetImageWithDescs0((float)ValueScrollBar.Value);
    }

    private Model3 _model = null!;

    private Random _random = null!;
}