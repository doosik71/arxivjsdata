# Foveated Instance Segmentation

Hongyi Zeng, Wenxuan Liu, Tianhua Xia, Jinhui Chen, Ziyun Li, Sai Qian Zhang (2025)

## 🧩 Problem to Solve

본 논문은 증강 현실(AR) 및 가상 현실(VR) 환경에서 필수적인 Instance Segmentation의 높은 계산 비용 문제를 해결하고자 한다. AR/VR 기기는 몰입감 있는 경험을 제공하기 위해 고해상도 이미지(예: Meta Ray-Ban 글래스의 1440P 비디오)를 캡처하지만, 이를 실시간으로 처리하기에는 리소스 제약이 심한 엣지 디바이스의 계산 능력이 부족하다.

이러한 고해상도 데이터 처리는 심각한 처리 지연(Latency)을 초래하며, 이는 실시간 상호작용을 방해하여 사용자 경험을 저하시킨다. 특히 기존의 Video Instance Segmentation 방식은 여러 프레임을 묶어 처리함으로써 temporal correlation을 활용하지만, 모든 프레임이 수집될 때까지 처리를 시작할 수 없으므로 실시간 응답성이 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 인간의 시각적 특성인 Foveated Vision(중심와 시각)에 착안하여, 사용자의 시선이 머무는 관심 영역인 Instance of Interest (IOI)만을 선택적으로 세그멘테이션함으로써 계산 부하를 획기적으로 줄이고 실시간 성능을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 눈이 모든 영역을 동일한 해상도로 보는 것이 아니라, 시선이 집중된 좁은 영역만 정밀하게 인식하고 주변부는 낮게 인식한다는 점을 딥러닝 아키텍처에 적용한 것이다.

1.  **FSNet (Foveated Segmentation Network)**: 고해상도 입력 이미지와 실시간 시선 위치(Gaze Location)를 입력으로 받아, IOI에 대해서만 효율적으로 세그멘테이션을 수행하는 플러그 앤 플레이(Plug-and-play) 구조의 네트워크를 제안한다. 이는 기존의 다양한 세그멘테이션 네트워크와 결합 가능하다.
2.  **FovealSeg Framework**: FSNet을 기반으로 하여, 인간의 시선 패턴(Fixation 및 Saccade)과 프레임 간의 시간적 유사성을 활용한 전체 프레임워크를 제안한다. 시선이 급격히 변하는 Saccade 구간에서는 처리를 생략하고, 시선이 안정적인 Fixation 구간에서는 이전 결과를 재사용함으로써 중복 계산을 최소화한다.

## 📎 Related Works

### 인간의 시선 행동 및 특성
인간의 시각 활동은 크게 세 가지 모드로 나뉜다.
- **Fixation**: 시선이 한 점에 고정된 상태로, 이 영역의 시각적 예리함이 가장 높다.
- **Saccade**: 시선을 한 대상에서 다른 대상으로 빠르게 옮기는 도약 운동이다. 이 과정에서 시각 정보가 일시적으로 흐려지는 Saccadic Blur 현상이 발생하며, 뇌는 이 시기의 정보 감지를 약 75%까지 억제한다.
- **Smooth Pursuit**: 움직이는 대상을 부드럽게 따라가는 움직임이다.

### 기존 접근 방식 및 한계
- **Saliency-based Downsampling**: 기존의 LTD나 LZU 같은 연구들은 학습 가능한 다운샘플링을 통해 효율성을 높이려 했다. 그러나 이들은 이미지 전체의 중요도를 학습하는 방식이며, 사용자의 실시간 의도(시선)를 직접적으로 반영하지 않는다.
- **Video Instance Segmentation**: temporal correlation을 위해 여러 프레임을 동시에 처리하지만, 이는 필연적으로 대기 시간을 발생시켜 실시간 AR/VR 요구 사항(50-100ms 이하의 지연 시간)을 충족하기 어렵다.

## 🛠️ Methodology

### 1. 이미지 샘플링 (Image Sampling)
입력 이미지 $F \in \mathbb{R}^{H \times W \times C}$를 저해상도 이미지 $\hat{F} \in \mathbb{R}^{h \times w \times C}$로 변환하기 위해 매핑 함수 $g_h(\cdot)$와 $g_w(\cdot)$를 사용한다.
$$\hat{F}[i,j] := F[g_h(i,j), g_w(i,j)]$$
본 논문에서는 Saliency DNN이 생성한 점수 맵 $D_\theta(i,j)$와 Gaussian kernel $k_\sigma$를 이용해 가중치 기반의 샘플링을 수행한다. 이는 시선이 집중된 영역의 샘플링 밀도를 높여 IOI 영역을 확대(Enlarge)하는 효과를 준다.

### 2. FSNet 아키텍처
FSNet은 시선 정보를 우선적으로 반영하는 구조를 가진다.

- **Gaze Map ($N$) 생성**: 시선 위치 $(u, v)$를 중심으로 정규화된 역거리를 계산하여 맵을 생성한다.
  $$N[i,j] = 1 - \frac{\sqrt{(i-u)^2 + (j-v)^2}}{d_{max}}$$
- **Saliency-guided Downsampling**: 입력 이미지 $F$와 시선 맵 $N$을 결합하여 Saliency DNN에 입력하고, 여기서 나온 점수 맵을 통해 저해상도 이미지 $\hat{F}$를 생성한다.
- **Dual-branch Segmentation**: 세그멘테이션 네트워크 $S$를 두 개의 브랜치로 나누어 효율성을 높였다.
    - $S_{seg}(\cdot)$: IOI 영역에 대한 이진 마스크 $Y_{bm} \in \mathbb{R}^{h \times w \times 1}$을 생성한다.
    - $S_{cls}(\cdot)$: IOI 내부 객체의 클래스를 분류하여 $Y_{cls} \in \mathbb{R}^{C \times 1}$를 생성한다.
    - 최종 결과 $Y_{cm}$은 두 출력의 외적(Outer product)으로 생성된다.

### 3. 손실 함수 및 학습 전략
- **Loss Function**: Dice loss와 가중치 Focal loss를 결합하여 사용한다. 특히 IOI가 매우 작은 경우(예: 칼, 냄비) 배경에 편향되는 문제를 해결하기 위해 IOI 영역과 비-IOI 영역의 면적 역수를 가중치로 적용한 Focal loss를 사용한다.
  $$\mathcal{L}_{tot} = \mathcal{L}_{dice}(Y'_{gt}, Y_{cm}) + \lambda \mathcal{L}_{focal}(Y'_{gt}, Y_{cm})$$
- **Alternative Training Strategy**: Saliency DNN과 Segmentation Network를 번갈아 학습시킨다. 먼저 Segmentation 부분을 고정하고 Saliency DNN을 학습시킨 후, 다시 Saliency DNN을 고정하고 Segmentation 부분을 미세 조정(Fine-tuning)한다.

### 4. FovealSeg 알고리즘 흐름
전체 시스템은 다음과 같은 로직으로 동작한다 (Algorithm 1 참조).
1. **Saccade 감지**: 현재 시선 $g_t$와 이전 시선 $g_{last}$의 거리 차이가 임계값 $\alpha$보다 크면 Saccade로 판단하고, 해당 프레임의 연산을 생략한다.
2. **프레임 유사도 검사**: 현재 프레임 $F_t$와 세그먼트 시작 프레임 $F_{init}$의 픽셀 차이가 임계값 $\beta$보다 크면 씬(Scene)이 변한 것으로 보고 FSNet을 재실행한다.
3. **시선 위치 확인**: 시선이 이전 세그멘테이션 마스크 $M_{last}$의 IOI 영역 내에 있다면 $M_{last}$를 재사용하고, 영역을 벗어났다면 FSNet을 통해 새로운 마스크를 생성한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Cityscapes, ADE20K, LVIS, Aria Everyday Activities.
- **비교 대상 (Baselines)**: 
    - **Avg**: 균일 샘플링(Uniform subsampling)을 사용하는 베이스라인.
    - **LTD**: 학습 가능한 다운샘플링 기반의 기존 연구.
    - **ND (No Downsample)**: 다운샘플링 없이 전체 해상도를 처리하는 방식.
    - **NS (No Skip)**: 다운샘플링은 하되, 프레임 스킵/재사용 없이 매번 처리하는 방식.
- **지표**: IoU 및 IoU' (전체 해상도 및 저해상도 버전에서의 성능).

### 주요 결과
- **정량적 성능**: FSNet은 모든 데이터셋에서 베이스라인(Avg) 대비 IoU가 최소 0.14 이상 향상되었다. 특히 SegFormer-B5를 백본으로 사용했을 때 가장 높은 성능(IoU 0.58, IoU' 0.59)을 보였다.
- **계산 효율성**: 
    - FovealSeg 프레임워크는 NS(No Skip) 대비 FLOPs를 최대 $1.96\times$ 감소시켰으며, ND(No Downsample) 대비 최대 $75\times$ 감소시켰다.
    - Jetson Orin NX 시뮬레이션 결과, SegFormer-B5의 지연 시간이 $1860\text{ms}$인 반면, FovealSeg는 $84\text{ms}$로 약 $20\times$ 이상의 속도 향상을 달성했다.
- **Ablation Study**: 
    - 다운샘플링 비율이 높아질수록(해상도가 낮아질수록) IoU는 감소하지만, 여전히 FSNet이 다른 방식보다 우월했다.
    - 시선 정보($u, v$)를 무작위 노이즈로 대체했을 때 IoU가 0.3 이상 급감하여 시선 가이드의 중요성이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 인간의 시각적 인지 특성을 딥러닝 파이프라인에 직접적으로 통합하여, 계산 효율성과 정확도라는 두 마리 토끼를 잡았다는 점에서 강점이 있다. 특히 단순한 다운샘플링이 아니라, 시선 위치를 기반으로 IOI 영역을 확대하여 샘플링하는 전략은 정보 손실을 최소화하면서 연산량을 줄이는 효과적인 방법이다.

또한, Saccade 구간의 처리 생략과 Fixation 구간의 결과 재사용이라는 temporal strategy를 도입하여, 하드웨어의 물리적 한계를 소프트웨어적 알고리즘으로 극복하려 한 점이 돋보인다.

다만, 본 시스템의 성능은 전적으로 실시간 시선 추적 시스템(Gaze Tracker)의 정확도와 지연 시간(5-10ms 수준)에 의존한다. 만약 시선 추적의 오차가 크거나 지연이 발생할 경우, 사용자가 바라보는 실제 대상과 세그멘테이션 영역이 일치하지 않는 mismatch 문제가 발생할 가능성이 있으며, 이에 대한 강건성(Robustness) 분석이 추가적으로 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 AR/VR 기기의 제한된 자원에서 실시간 Instance Segmentation을 수행하기 위해, 인간의 시선 데이터를 활용해 관심 영역(IOI)만 처리하는 **FSNet**과 시선 패턴 기반의 프레임워크인 **FovealSeg**를 제안하였다. 이 방법은 연산량을 획기적으로 줄여 지연 시간을 $1860\text{ms}$에서 $84\text{ms}$로 단축하면서도 높은 정확도를 유지하였으며, 이는 향후 고해상도 AR/VR 기기의 실시간 상호작용 시스템 구현에 핵심적인 역할을 할 것으로 기대된다.