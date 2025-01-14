using System;
using System.Drawing;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;


namespace WinFormsApp2
{
    public partial class Form1 : Form
    {

        private PictureBox PictureBox1;

        public Form1()
        {
            InitializeComponent();


            PictureBox1 = new PictureBox()
            {
                Location = new System.Drawing.Point(10, 10),
                Size = new System.Drawing.Size(800, 600),
                BorderStyle = BorderStyle.Fixed3D,
                SizeMode = PictureBoxSizeMode.Zoom
            };

            this.Controls.Add(PictureBox1); //this = Form1

            this.Load += Form1_Load;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            string filePath = "example.jpg";

            using (Mat Image = new Mat(filePath))
            {
                PictureBox1.Image = BitmapConverter.ToBitmap(Image);
            }

        }



    }
}
