using Ssz.AI.Models;

namespace Ssz.AI.ViewModels
{
    public partial class MainWindowViewModel : ViewModelBase
    {
        public MainWindowViewModel() 
        { 
            Model = new Model1();
        }

        public Model1 Model { get; }
    }
}
