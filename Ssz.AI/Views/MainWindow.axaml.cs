using Avalonia.Controls;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Ssz.Utils;

namespace Ssz.AI.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            // DO NOT WORKING
            //Title = ConfigurationHelper.GetValue<string>(Program.Host.Services.GetRequiredService<IConfiguration>(), Program.ConfigurationKey_Value, @"<none>");
        }
    }
}