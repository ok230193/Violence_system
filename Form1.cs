using System;
using System.Windows.Forms;
using OpenCvSharp.Extensions;
using OpenCvSharp;


namespace WinFormsApp3
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private System.Windows.Forms.Timer timer;
        private PictureBox pictureBox1;


        public Form1()
        {
            InitializeComponent();

            pictureBox1 = new PictureBox
            {
                Location = new System.Drawing.Point(10,10),
                Size = new System.Drawing.Size(800,600),
                SizeMode = PictureBoxSizeMode.Zoom
            };

            this.Controls.Add(pictureBox1);

            this.Load += Form1_Load;

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            capture = new VideoCapture(0);
            timer = new System.Windows.Forms.Timer
            {
                Interval = 30
            };

            timer.Tick += Timer_Tick;
            timer.Start();
        }


        private void Timer_Tick(object sender, EventArgs e)
        {
            using (Mat frame = new Mat())
            { 
                capture.Read(frame);

                //グレースケール変換
                Mat grayFrame = new Mat();
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                //エッジ処理
                Mat edgeFrame = new Mat();
                Cv2.Canny(grayFrame, edgeFrame, 100, 200);

                pictureBox1.Image = BitmapConverter.ToBitmap(edgeFrame);

            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            timer.Stop();
            capture.Release();
            capture.Dispose();
        }

    }
}
