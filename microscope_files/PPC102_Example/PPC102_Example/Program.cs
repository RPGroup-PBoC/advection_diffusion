using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.Benchtop.PrecisionPiezoCLI;
using Thorlabs.MotionControl.GenericPiezoCLI.Settings;
using Thorlabs.MotionControl.GenericPiezoCLI.Piezo;

namespace PPC102_Example
{
    class Program
    {
        static void Main(string[] args)
        {
            String serialNumber = "95000025";
            SimulationManager.Instance.InitializeSimulations();

            try
            { DeviceManagerCLI.BuildDeviceList(); }
            catch (Exception ex) { return; }

            BenchtopPrecisionPiezo ppc = BenchtopPrecisionPiezo.CreateBenchtopPiezo(serialNumber);

            PrecisionPiezoChannel channel = ppc.GetChannel(1);
            channel.Connect(serialNumber);
            channel.WaitForSettingsInitialized(5000);
            channel.StartPolling(50);
            channel.EnableDevice();


            channel.SetOutputVoltage(20);

            Decimal newVolts = channel.GetOutputVoltage();
            Console.WriteLine("Voltage set to {0}", newVolts);

            channel.SetPositionControlMode(PiezoControlModeTypes.CloseLoop);
            System.Threading.Thread.Sleep(200);
            Console.WriteLine("Loop Model Changed to: {0}", channel.GetPositionControlMode());

            channel.StopPolling();
            ppc.Disconnect(true);

            SimulationManager.Instance.UninitializeSimulations();
        }
    }
}
