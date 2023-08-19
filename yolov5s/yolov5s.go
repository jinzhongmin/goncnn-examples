package main

import (
	"fmt"
	"image"
	"math"
	"sort"
	"time"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/parallel"
	"github.com/anthonynsimon/bild/transform"
	"github.com/fogleman/gg"
	"github.com/jinzhongmin/goffi/pkg/c"
	"github.com/jinzhongmin/goncnn/pkg/ncnn"
	"github.com/jinzhongmin/usf"
)

// init ncnn library by shared library file
func init() {
	//download from ncnn release page
	ncnn.InitLib("./ncnn.dll", c.ModeLazy)
}

func main() {

	//init ncnn
	var ex *ncnn.Extractor
	{
		opt := ncnn.CreateOption()
		opt.SetUseVulkanCompute(true)
		defer opt.Destroy()

		net := ncnn.CreateNet()
		defer net.Destroy()
		net.SetOption(opt)

		//load model,  refer to readme.md-> ## model convert
		net.LoadParam("./yolov5s.ncnn.param")
		net.LoadModel("./yolov5s.ncnn.bin")

		ex = ncnn.CreateExtractor(net)
		defer ex.Destroy()
		ex.SetOption(opt)
	}

	//init input image
	var srcImg image.Image
	var srcW, srcH int
	var imgBlob []float32 // input data
	{
		srcImg, _ = imgio.Open("bus.jpg")
		srcW, srcH = srcImg.Bounds().Dx(), srcImg.Bounds().Dy()

		imgBlob = Blob(transform.Resize(srcImg, 640, 640, transform.NearestNeighbor),
			[3]float32{0, 0, 0}, [3]float32{1.0 / 255, 1.0 / 255, 1.0 / 255})
	}

	//extract and get output data
	var outData []float32
	{
		alc := ncnn.CreatePoolAllocator()
		defer alc.Destroy()

		//create input mat
		inMat := ncnn.CreateMatExternal3D(640, 640, 3, &imgBlob[0], alc)
		defer inMat.Destroy()

		//input
		ex.Input("in0", inMat)

		t1 := time.Now()

		//Extract
		outMat := ex.Extract("out0")
		defer outMat.Destroy()

		dt := time.Since(t1)
		fmt.Printf("extrace: %s  //The first inference takes a little longer\n", dt)

		//get output data
		w := outMat.GetW()
		h := outMat.GetH()
		l := w * h
		outData = *(*[]float32)(usf.Slice(outMat.GetData(), uint64(l)))
	}

	//threshold control
	var score_thre float32 = 0.8
	var iou_thre float32 = 0.4

	//get result
	var results []*Result
	{
		results_all := make([]*Result, 25200)
		parallel.Line(25200, func(start, end int) {
			for r := start; r < end; r++ {
				p := outData[r*85 : r*85+85]
				if p[4] < 0.45 {
					continue
				}
				cx := float64(p[0])
				cy := float64(p[1])
				w := float64(p[2])
				h := float64(p[3])
				p = outData[r*85+5 : r*85+85]
				var bestVal float32 = 0.0
				bestIdx := 0
				for i := range p {
					if p[i] <= bestVal {
						continue
					}
					bestIdx = i
					bestVal = p[i]
				}
				if bestVal <= score_thre {
					continue
				}

				x0 := int(cx - w/2)
				y0 := int(cy - h/2)
				x1 := x0 + int(w)
				y1 := y0 + int(h)

				results_all[r] = &Result{
					rect:  image.Rect(x0, y0, x1, y1),
					score: bestVal,
					class: bestIdx,
				}
			}
		})

		results_no_nil := NewResults()
		for i := 0; i < 25200; i++ {
			if results_all[i] == nil {
				continue
			}
			results_no_nil = append(results_no_nil, results_all[i])
		}
		results = results_no_nil.Nms(iou_thre)
	}

	//output image
	{
		ctx := gg.NewContextForImage(srcImg)
		ctx.SetLineWidth(2)
		for _, r := range results {
			x0 := float64(r.rect.Min.X) / 640 * float64(srcW)
			x1 := float64(r.rect.Max.X) / 640 * float64(srcW)
			y0 := float64(r.rect.Min.Y) / 640 * float64(srcH)
			y1 := float64(r.rect.Max.Y) / 640 * float64(srcH)

			//control rectangle boundaries in the picture
			x0, y0 = math.Max(0, x0), math.Max(0, y0)
			x1, y1 = math.Min(x1, float64(srcW)), math.Min(y1, float64(srcH))

			ctx.SetHexColor("#00FF88")
			ctx.DrawRectangle(x0, y0, x1-x0, y1-y0)
			ctx.Stroke()
			class := className[r.class]
			ctx.SetHexColor("#ff0000")
			ctx.DrawString(class, (x0+x1)/2, y0)

		}
		ctx.SavePNG("output.png")
	}
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

type Result struct {
	rect  image.Rectangle
	class int
	score float32
}

func (r Results) Len() int           { return len(r) }
func (r Results) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r Results) Less(i, j int) bool { return r[i].score < r[j].score }

type Results []*Result

func NewResults() Results {
	return make([]*Result, 0)
}

func (rs Results) Nms(iou_thre float32) []*Result {
	sort.Sort(Results(rs))
	rnms := make([]*Result, 0)
	if len(rs) == 0 {
		return rnms
	}
	rnms = append(rnms, rs[0])

	for i := 1; i < len(rs); i++ {
		tocheck, del := len(rnms), false
		for j := 0; j < tocheck; j++ {
			currIOU := iou(rs[i].rect, rnms[j].rect)
			if currIOU > iou_thre {
				del = true
				break
			}
		}
		if !del {
			rnms = append(rnms, rs[i])
		}
	}
	return rnms
}
func iou(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

var className = map[int]string{
	0:  "person",
	1:  "bicycle",
	2:  "car",
	3:  "motorcycle",
	4:  "airplane",
	5:  "bus",
	6:  "train",
	7:  "truck",
	8:  "boat",
	9:  "traffic light",
	10: "fire hydrant",
	11: "stop sign",
	12: "parking meter",
	13: "bench",
	14: "bird",
	15: "cat",
	16: "dog",
	17: "horse",
	18: "sheep",
	19: "cow",
	20: "elephant",
	21: "bear",
	22: "zebra",
	23: "giraffe",
	24: "backpack",
	25: "umbrella",
	26: "handbag",
	27: "tie",
	28: "suitcase",
	29: "frisbee",
	30: "skis",
	31: "snowboard",
	32: "sports ball",
	33: "kite",
	34: "baseball bat",
	35: "baseball glove",
	36: "skateboard",
	37: "surfboard",
	38: "tennis racket",
	39: "bottle",
	40: "wine glass",
	41: "cup",
	42: "fork",
	43: "knife",
	44: "spoon",
	45: "bowl",
	46: "banana",
	47: "apple",
	48: "sandwich",
	49: "orange",
	50: "broccoli",
	51: "carrot",
	52: "hot dog",
	53: "pizza",
	54: "donut",
	55: "cake",
	56: "chair",
	57: "couch",
	58: "potted plant",
	59: "bed",
	60: "dining table",
	61: "toilet",
	62: "tv",
	63: "laptop",
	64: "mouse",
	65: "remote",
	66: "keyboard",
	67: "cell phone",
	68: "microwave",
	69: "oven",
	70: "toaster",
	71: "sink",
	72: "refrigerator",
	73: "book",
	74: "clock",
	75: "vase",
	76: "scissors",
	77: "teddy bear",
	78: "hair drier",
	79: "toothbrush",
}
