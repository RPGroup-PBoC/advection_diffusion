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
//using File;
//using FileInfo;
//using Task;

namespace PatternGridLines
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

            // Get available Benchtop Precision Piezo Motors and check our serial number is correct - by using the device prefix
            // (i.e. the device prefix is 95)
            List<string> benchtopSerialNumbers = DeviceManagerCLI.GetDeviceList(BenchtopPrecisionPiezo.DevicePrefix95);
            if (!benchtopSerialNumbers.Contains(ppcSerialNo))
            {
                // The requested serial number is not a PPC102 or is not connected
                Console.WriteLine("{0} is not a valid serial number", ppcSerialNo);
                Console.ReadKey();
                return;
            }

            // Create the BenchtopPrecisionPiezo device
            BenchtopPrecisionPiezo ppc = BenchtopPrecisionPiezo.CreateBenchtopPiezo(ppcSerialNo);
            if (ppc == null)
            {
                // An error occured
                Console.WriteLine("{0} is not a BenchtopPrecisionPiezo", ppcSerialNo);
                Console.ReadKey();
                return;
            }

            PrecisionPiezoChannel channely = ppc.GetChannel(1);
            PrecisionPiezoChannel channelx = ppc.GetChannel(2);

            // Open a connection to a channel on the device.
            try
            {
                Console.WriteLine("Opening X and Y PPC channels");
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
                    Console.WriteLine(channely.IsSettingsInitialized());
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
                    Console.WriteLine(channelx.IsSettingsInitialized());
                }
                catch (Exception)
                {
                    Console.WriteLine("Channel 2 (x-axis) settings failed to initialized");
                }
            }

            // Start the device polling
            // The polling loop requests regular status requests to the motor to ensure the program keeps track of the device.
            channely.StartPolling(250);
            // Needs a delay so that the current enabled state can be obtained
            Thread.Sleep(500);
            // Enable the channel otherwise any move is ignored 
            channely.EnableDevice();
            // Needs a delay to give time for the device to be enabled
            Thread.Sleep(500);

            // Same for channel 2
            channelx.StartPolling(250);
            Thread.Sleep(500);
            channelx.EnableDevice();
            Thread.Sleep(500);

            // display info about device
            DeviceInfo chany_Info = channely.GetDeviceInfo();
            Console.WriteLine("Gimbal Mount Y {0} = {1}", chany_Info.SerialNumber, chany_Info.Name);

            DeviceInfo chanx_Info = channelx.GetDeviceInfo();
            Console.WriteLine("Gimbal Mount X {0} = {1}", chanx_Info.SerialNumber, chanx_Info.Name);

            channely.SetPositionControlMode(PiezoControlModeTypes.CloseLoop);
            channelx.SetPositionControlMode(PiezoControlModeTypes.CloseLoop);
            Thread.Sleep(200);
            Console.WriteLine("Loop Model Changed to: {0}", channely.GetPositionControlMode());
            Console.WriteLine("Loop Model Changed to: {0}", channelx.GetPositionControlMode());

            // Set neutral position
            channely.SetPosition(0);
            channelx.SetPosition(0);


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
                Console.WriteLine("Opening device {0}", kcubeSerialNo);
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
            kcube.StartPolling(250);
            // Needs a delay so that the current enabled state can be obtained
            Thread.Sleep(500);
            // Enable the channel otherwise any move is ignored
            kcube.EnableDevice();
            // Needs a delay to give time for the device to be enabled
            Thread.Sleep(500);

            // Call LoadMotorConfiguration on the device to initialize the 
            // DeviceUnitConverter object required for real worl unit parameters
            // - loads configuration information into channel
            MotorConfiguration kcubeConfiguration = kcube.LoadMotorConfiguration(kcubeSerialNo);

            // Not used directly in example but illustrates how to obtain device settings
            KCubeBrushlessMotorSettings currentKCubeSettings = kcube.MotorDeviceSettings as KCubeBrushlessMotorSettings;

            // Display info about device
            DeviceInfo kcubeInfo = kcube.GetDeviceInfo();
            Console.WriteLine("KCube {0} = {1}", kcubeInfo.SerialNumber, kcubeInfo.Name);

            Home_Method1(kcube);
            bool homed = kcube.Status.IsHomed;

            VelocityParameters velPars = kcube.GetVelocityParams();
            velPars.MaxVelocity = 1800;
            kcube.SetVelocityParams(velPars);

            Move_Method1(kcube, (decimal)104.5);

            Decimal newRotationPos = kcube.Position;
            Console.WriteLine("Rotation mount moved to {0}", newRotationPos);

            // Set initial position
            decimal posInit = -1 * (decimal)10;
            int totalSteps = 5;
            Decimal peakPositionRange = 20;

            channelx.SetPosition(posInit);
            Thread.Sleep(500);
            Console.WriteLine(channelx.GetPosition());

            bool keep_active = true;

            while (keep_active)
            {
                Move_Method1(kcube, (decimal)104.5);
                channely.SetPosition(0);
                Thread.Sleep(150);

                for (short reps = 0; reps < 2; reps++)
                {
                    channelx.SetPosition(posInit);

                    Thread.Sleep(150);

                    for (short nX = 0; nX < totalSteps; nX++)
                    {
                        Decimal newXPos = posInit + peakPositionRange * ((Decimal)nX + 1) / totalSteps;

                        if (newXPos > channelx.GetMaxTravel() | newXPos < -1 * channelx.GetMaxTravel())
                        {
                            Console.WriteLine("Position is outside the limits of the mirror mount range.");
                            return;
                        }
                        else
                        {
                            channelx.SetPosition(newXPos); // Arg in mrad
                            Thread.Sleep(150);
                        }
                    }
                }

                Move_Method1(kcube, (decimal)14.5);
                channelx.SetPosition(0);
                Thread.Sleep(150);

                for (short reps = 0; reps < 2; reps++)
                {
                    channely.SetPosition(posInit);

                    Thread.Sleep(150);

                    for (short nY = 0; nY < totalSteps; nY++)
                    {
                        Decimal newYPos = posInit + peakPositionRange * ((Decimal)nY + 1) / totalSteps;

                        if (newYPos > channely.GetMaxTravel() | newYPos < -1 * channely.GetMaxTravel())
                        {
                            Console.WriteLine("Position is outside the limits of the mirror mount range.");
                            return;
                        }
                        else
                        {
                            channely.SetPosition(newYPos); // Arg in mrad
                            Thread.Sleep(150);
                        }
                    }
                }
            }

            // Return to neutral position
            channely.SetPosition(0);
            channelx.SetPosition(0);
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

        private static bool _taskComplete;
        private static ulong _taskID;

        public static void CommandCompleteFunction(ulong taskID)
        {
            if ((_taskID > 0) && (_taskID == taskID))
            {
                _taskComplete = true;
            }
        }

    }
}
