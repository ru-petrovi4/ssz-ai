using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Ssz.AI.ViewModels;
using System.Collections.ObjectModel;
using System.Linq;

namespace Ssz.AI;

public partial class Sliders : UserControl
{
    public Sliders()
    {
        InitializeComponent();
    }
}

public class SlidersItem : ViewModelBase
{
    private double _value;

    /// <summary>
    /// Значение слайдера (например, уровень громкости в этом диапазоне)
    /// </summary>
    public double Value
    {
        get => _value;
        set
        {
            if (_value != value)
            {
                _value = value;
                OnPropertyChanged();
            }
        }
    }

    /// <summary>
    ///     Метка
    /// </summary>
    public string Label { get; set; } = @"";
}

public class SlidersViewModel : ViewModelBase
{
    public SlidersViewModel()
    {
        foreach (int i in Enumerable.Range(0, 22))
        {
            SlidersItems.Add(new SlidersItem()
            {
                Value = 0.0,
                Label = i.ToString(),
            });
        }
    }

    /// <summary>
    /// Коллекция диапазонов эквалайзера
    /// </summary>
    public ObservableCollection<SlidersItem> SlidersItems { get; } = new ObservableCollection<SlidersItem>();
}
