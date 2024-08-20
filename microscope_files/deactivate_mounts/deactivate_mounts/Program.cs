using System;
using Thorlabs.MotionControl.Benchtop.PrecisionPiezoCLI;
using Thorlabs.MotionControl.KCube.BrushlessMotorCLI;

namespace deactivate_mounts
{
    class Program
    {
        static void Main(string[] args)
        {
            string ppcSerialNo = "95000025";
            string kcubeSerialNo = "28251566";

            // Create the BenchtopPrecisionPiezo device
            BenchtopPrecisionPiezo ppc = BenchtopPrecisionPiezo.CreateBenchtopPiezo(ppcSerialNo);
            PrecisionPiezoChannel channely = ppc.GetChannel(1);
            PrecisionPiezoChannel channelx = ppc.GetChannel(2);

            // Create the device - KCube Brushhless motor
            KCubeBrushlessMotor kcube = KCubeBrushlessMotor.CreateKCubeBrushlessMotor(kcubeSerialNo);
            kcube.Connect(kcubeSerialNo);

            Console.WriteLine("Returning to neutral position");
            channely.SetPosition(0);
            channelx.SetPosition(0);

            channely.StopPolling();
            channelx.StopPolling();
            ppc.Disconnect(true);

            kcube.StopPolling();
            kcube.Disconnect(true);
        }
    }
}
