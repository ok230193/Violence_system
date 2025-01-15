using System;
using System.Drawing;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace WinFormsApp1
{
    public partial class Form1 : Form
    {
        private VideoCapture capture; // カメラキャプチャ用変数
        private System.Windows.Forms.Timer timer; // フレーム更新用タイマー用変数
        private PictureBox pictureBox1; // 映像表示用PictureBox用変数

        public Form1()
        {
            InitializeComponent();

            // PictureBoxの設定
            pictureBox1 = new PictureBox
            {
                Location = new System.Drawing.Point(1, 1),
                Size = new System.Drawing.Size(900, 600),
                SizeMode = PictureBoxSizeMode.Zoom
            };
            Controls.Add(pictureBox1);

            // Loadイベントでカメラを初期化　　Loadで画面表示(表示させるために先に定義)、Form1_Loadで表示した後の行動(動き)=(表示と動きの処理は別で考える)
            Load += Form1_Load;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            capture = new VideoCapture(0); // 内蔵カメラを初期化

            // タイマーを設定して映像を更新
            timer = new System.Windows.Forms.Timer(); // Windows Forms用のタイマー
            timer.Interval = 30; // フレーム更新間隔（約30FPS）
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            using (Mat frame = new Mat())//usingを使うことで{}内のみでframeを使えるようにし、{}を抜けると消える。
            {
                capture.Read(frame); // カメラからフレームを取得
                pictureBox1.Image = BitmapConverter.ToBitmap(frame); // PictureBoxに表示
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e) //終了時に自動で起動
        {
            // リソースを解放
            timer.Stop();
            capture.Release();
            capture.Dispose();
        }
    }
}