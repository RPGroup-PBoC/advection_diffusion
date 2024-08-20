using System;
using System.Collections.Generic;
using System.Threading;
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.GenericPiezoCLI.Piezo;
using Thorlabs.MotionControl.Benchtop.PrecisionPiezoCLI;
using Thorlabs.MotionControl.GenericMotorCLI;
using Thorlabs.MotionControl.GenericMotorCLI.ControlParameters;
using Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor;
using Thorlabs.MotionControl.GenericMotorCLI.KCubeMotor;
using Thorlabs.MotionControl.GenericMotorCLI.Settings;
using Thorlabs.MotionControl.KCube.BrushlessMotorCLI;

namespace rotate_90
{
    class Program
    {
        static void Main(string[] args)
        {
            string ppcSerialNo = "95000025";
            string kcubeSerialNo = "28251566";
            try
            {
                // Tell the device manager to get the list of all devices connected to the computer, such as the controller
                DeviceManagerCLI.BuildDeviceList();
            }
            catch (Exception ex)
            {
                // An error occurred - see ex for details
                Console.WriteLine("Exception raised by BuildDeviceList {0}", ex);
                Console.ReadKey();
                return;
            }

            // Create the BenchtopPrecisionPiezo device
            BenchtopPrecisionPiezo ppc = BenchtopPrecisionPiezo.CreateBenchtopPiezo(ppcSerialNo);
            PrecisionPiezoChannel channely = ppc.GetChannel(1);
            PrecisionPiezoChannel channelx = ppc.GetChannel(2);

            // Open a connection to a channel on the device.
            try
            {
                channely.Connect(ppcSerialNo);
                channelx.Connect(ppcSerialNo);
            }
            catch (Exception)
            {
                // Connection failed
                Console.WriteLine("Failed to open device {0}", ppcSerialNo);
                Console.ReadKey();
                return;
            }

            // Get the correct channel - channel 1
            if (channely == null)
            {
                Console.WriteLine("Channel 1 (y-axis) unavailable {0}", ppcSerialNo);
                Console.ReadKey();
                return;
            }

            if (channelx == null)
            {
                Console.WriteLine("Channel 2 (x-axis) unavailable {0}", ppcSerialNo);
                Console.ReadKey();
                return;
            }

            // Wait for the device settings to initialize - timeout 5000ms
            if (!channely.IsSettingsInitialized())
            {
                try
                {
                    channely.WaitForSettingsInitialized(5000);
                }
                catch (Exception)
                {
                    Console.WriteLine("Channel 1 (y-axis) settings failed to initialize");
                }
            }

            if (!channelx.IsSettingsInitialized())
            {
                try
                {
                    channelx.WaitForSettingsInitialized(5000);
                }
                catch (Exception)
                {
                    Console.WriteLine("Channel 2 (x-axis) settings failed to initialized");
                }
            }

            // Start the device polling
            // The polling loop requests regular status requests to the motor to ensure the program keeps track of the device.
            channely.StartPolling(150);
            // Needs a delay so that the current enabled state can be obtained
            Thread.Sleep(100);
            // Enable the channel otherwise any move is ignored 
            channely.EnableDevice();
            // Needs a delay to give time for the device to be enabled
            Thread.Sleep(100);

            // Same for channel 2
            channelx.StartPolling(150);
            Thread.Sleep(100);
            channelx.EnableDevice();
            Thread.Sleep(100);

            // Verify for KCube Brushless Motor
            List<string> kbmSerialNumbers = DeviceManagerCLI.GetDeviceList(KCubeBrushlessMotor.DevicePrefix);
            if (!kbmSerialNumbers.Contains(kcubeSerialNo))
            {
                // The requested serial number is not a KBD101 or is not connected
                Console.WriteLine("{0} is not a valid serial number", kcubeSerialNo);
                Console.ReadKey();
                return;
            }

            // Create the device - KCube Brushhless motor
            KCubeBrushlessMotor kcube = KCubeBrushlessMotor.CreateKCubeBrushlessMotor(kcubeSerialNo);
            if (kcube == null)
            {
                // An error occured
                Console.WriteLine("{0} is not a KCubeBrushlessMotor", kcubeSerialNo);
                Console.ReadKey();
                return;
            }

            // Open a connection to the device.
            try
            {
                kcube.Connect(kcubeSerialNo);
            }
            catch (Exception)
            {
                // Connection failed
                Console.WriteLine("Failed to open device {0}", kcubeSerialNo);
                Console.ReadKey();
                return;
            }

            // Wait for the device settings to initialize - timeout 5000ms
            if (!kcube.IsSettingsInitialized())
            {
                try
                {
                    kcube.WaitForSettingsInitialized(5000);
                }
                catch (Exception)
                {
                    Console.WriteLine("Settings failed to initialize");
                }
            }

            // Start polling the device
            kcube.StartPolling(150);
            // Needs a delay so that the current enabled state can be obtained
            Thread.Sleep(100);
            // Enable the channel otherwise any move is ignored
            kcube.EnableDevice();
            // Needs a delay to give time for the device to be enabled
            Thread.Sleep(100);

            // Call LoadMotorConfiguration on the device to initialize the 
            // DeviceUnitConverter object required for real worl unit parameters
            // - loads configuration information into channel
            MotorConfiguration kcubeConfiguration = kcube.LoadMotorConfiguration(kcubeSerialNo);

            // Not used directly in example but illustrates how to obtain device settings
            KCubeBrushlessMotorSettings currentKCubeSettings = kcube.MotorDeviceSettings as KCubeBrushlessMotorSettings;

            // Display info about device
            DeviceInfo kcubeInfo = kcube.GetDeviceInfo();

            //Home_Method1(kcube);

            Move_Method1(kcube, (decimal)11.5);

            Decimal newRotationPos = kcube.Position;

            // Set initial position
            decimal posInit = -1 * (decimal)10;

            channelx.SetPosition(0);
            channely.SetPosition(0);

        }
        public static void Home_Method1(IGenericAdvancedMotor device)
        {
            try
            {
                Console.WriteLine("Homing device");
                device.Home(60000);
            }
            catch (Exception)
            {
                Console.WriteLine("Failed to home device");
                Console.ReadKey();
                return;
            }
            Console.WriteLine("Device Homed");
        }

        public static void Move_Method1(IGenericAdvancedMotor device, decimal position)
        {
            try
            {
                Console.WriteLine("Moving Device to {0}", position);
                device.MoveTo(position, 60000);
            }
            catch (Exception)
            {
                Console.WriteLine("Failed to move to position");
                Console.ReadKey();
                return;
            }
            Console.WriteLine("Device Moved to {0}", position);
        }
    }
}