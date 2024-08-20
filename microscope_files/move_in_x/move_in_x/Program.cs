using System;
using System.Collections.Generic;
using System.Threading;
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.GenericPiezoCLI.Piezo;
using Thorlabs.MotionControl.Benchtop.PrecisionPiezoCLI;
using Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor;

namespace move_in_x
{
    class Program
    {
        static void Main(string[] args)
        {
            string ppcSerialNo = "95000025";
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

            // Set initial position
            decimal posInit = -1 * (decimal)10;
            int totalSteps = 5;
            Decimal peakPositionRange = 20;

            for (short nX = 0; nX < totalSteps; nX++)
            {
                Decimal newXPos = posInit + peakPositionRange * ((Decimal)(nX + 3) % totalSteps) / totalSteps;

                if (newXPos > channelx.GetMaxTravel() | newXPos < -1 * channelx.GetMaxTravel())
                {
                    Console.WriteLine("Position is outside the limits of the mirror mount range.");
                    return;
                }
                else
                {
                    channelx.SetPosition(newXPos); // Arg in mrad
                    Thread.Sleep(200);
                }
            }

            channely.StopPolling();
            channelx.StopPolling();
            ppc.Disconnect(true);
        }
    }
}