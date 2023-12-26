package main

import (
	"image"
	"image/color"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/parallel"
	"github.com/anthonynsimon/bild/transform"
	"github.com/fogleman/gg"
	"github.com/jinzhongmin/goffi/pkg/c"
	"github.com/jinzhongmin/goncnn/pkg/ncnn"
	"github.com/jinzhongmin/usf"
)

func init() {
	ncnn.InitLib("./ncnn.dll", c.ModeLazy)
}

func main() {
	var ex *ncnn.Extractor
	{
		opt := ncnn.CreateOption()
		opt.SetUseVulkanCompute(true)
		defer opt.Destroy()

		net := ncnn.CreateNet()
		defer net.Destroy()
		net.SetOption(opt)

		net.LoadParam("./deeplabv3-plus.ncnn.param")
		net.LoadModel("./deeplabv3-plus.ncnn.bin")

		ex = ncnn.CreateExtractor(net)
		defer ex.Destroy()
		ex.SetOption(opt)
	}

	var srcImg image.Image
	var srcW, srcH int
	var imgBlob []float32 // input data
	{
		srcImg, _ = imgio.Open("street.jpg")
		srcW, srcH = srcImg.Bounds().Dx(), srcImg.Bounds().Dy()
		imgBlob = Blob(transform.Resize(srcImg, 512, 512, transform.NearestNeighbor),
			[3]float32{0, 0, 0}, [3]float32{1.0 / 255, 1.0 / 255, 1.0 / 255})
	}

	var outData []float32
	{
		alc := ncnn.CreatePoolAllocator()
		defer alc.Destroy()

		//create input mat
		inMat := ncnn.CreateMatExternal3D(512, 512, 3, &imgBlob[0], alc)
		defer inMat.Destroy()

		//input
		ex.Input("images", inMat)

		//Extract
		outMat := ex.Extract("output")
		defer outMat.Destroy()

		//get output data
		w := outMat.GetW()
		h := outMat.GetH()
		c := outMat.GetC() //class num
		l := w * h * c
		outData = *(*[]float32)(usf.Slice(outMat.GetData(), uint64(l)))
	}

	mask := image.NewRGBA(image.Rect(0, 0, 512, 512))
	for row := 0; row < 512; row++ {
		for col := 0; col < 512; col++ {
			max := classGetMax(outData) //max is dst type idx -> className[max]
			v := classColor[max]        //v is define type color
			mask.SetRGBA(col, row, color.RGBA{v[0], v[1], v[2], 180})
			classIdxAdd()
		}
	}
	imgio.Save("mask.png", mask, imgio.PNGEncoder())

	gc := gg.NewContextForImage(srcImg)
	gc.DrawImage(transform.Resize(mask, srcW, srcH, transform.NearestNeighbor), 0, 0)

	imgio.Save("mask_with_src.png", gc.Image(), imgio.PNGEncoder())
}

func Blob(im *image.RGBA, mean [3]float32, scale [3]float32) []float32 {
	rows := (*im).Bounds().Dy()
	cols := (*im).Bounds().Dx()
	frame := rows * cols
	frame2 := frame * 2

	rgb := make([]float32, rows*cols*3)
	s0 := scale[0]
	s1 := scale[1]
	s2 := scale[2]
	parallel.Line(rows, func(start, end int) {
		idx := start * rows

		for row := start; row < end; row++ {
			for col := 0; col < cols; col++ {
				r, g, b, _ := (*im).At(col, row).RGBA()
				rgb[idx] = (float32(r>>8) - mean[0]) * s0
				rgb[idx+frame] = (float32(g>>8) - mean[1]) * s1
				rgb[idx+frame2] = (float32(b>>8) - mean[2]) * s2
				idx += 1
			}
		}
	})
	return rgb
}

var classIdx = []int{}

func init() {
	classIdx = make([]int, 21, 21)
	for i := 0; i < 21; i++ {
		classIdx[i] = i * 512 * 512
	}
}
func classIdxAdd() {
	for i := 0; i < 21; i++ {
		classIdx[i] += 1
	}
}
func classGetMax(data []float32) int {
	max := float32(0)
	maxIdx := 0
	for i := 0; i < 21; i++ {
		v := data[classIdx[i]]
		if v > max {
			maxIdx = i
			max = v
		}
	}
	return maxIdx
}

var className = []string{
	"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
}

var classColor = [][]byte{
	{0, 0, 0},
	{128, 0, 0},
	{0, 128, 0},
	{128, 128, 0},
	{0, 0, 128},
	{128, 0, 128},
	{0, 128, 128},
	{128, 128, 128},
	{64, 0, 0},
	{192, 0, 0},
	{64, 128, 0},
	{192, 128, 0},
	{64, 0, 128},
	{192, 0, 128},
	{64, 128, 128},
	{192, 128, 128},
	{0, 64, 0},
	{128, 64, 0},
	{0, 192, 0},
	{128, 192, 0},
	{0, 64, 128},
	{128, 64, 12},
}
