// The shader code is converted to use in Unity3D from 
// "Displacement with Dispersion" created by cornusammonis on Shadertoy(https://www.shadertoy.com/view/4ldGDB)

Shader "Custom/Dispersion" {
	Properties{
		iChannel1("Albedo (RGB)", 2D) = "white" {}
		iChannel2("Albedo (RGB)", 2D) = "white" {}
	}
	SubShader{
	Tags{ "RenderType" = "Opaque" }
	LOD 200
	Pass{
	CGPROGRAM
	#pragma vertex vert_img
	#pragma fragment frag

	#include "UnityCG.cginc"

	sampler2D _bufferB;
	sampler2D iChannel1;

	#define _G0 0.25
	#define _G1 0.125
	#define _G2 0.0625
	#define W0 -3.0
	#define W1 0.5
	#define TIMESTEP 0.1
	#define ADVECT_DIST 2.0
	#define DV 0.70710678

	float nl(float x) {
		return 1.0 / (1.0 + exp(W0 * (W1 * x - 0.5)));
	}

	float4 gaussian(float4 x, float4 x_nw, float4 x_n, float4 x_ne, float4 x_w, float4 x_e, float4 x_sw, float4 x_s, float4 x_se) {
		return _G0*x + _G1*(x_n + x_e + x_w + x_s) + _G2*(x_nw + x_sw + x_ne + x_se);
	}

	bool reset() {
		// for now
		float reset = tex2D(iChannel1, float2(32.5 / 256.0, 0.5));

		return reset.x > 0.5;
	}

	float2 normz(float2 x) {
		return x == float2(0.0, 0.0) ? float2(0.0, 0.0) : normalize(x);
	}

	float4 advect(float2 ab, float2 vUv, float2 step) {

		float2 aUv = vUv - ab * ADVECT_DIST * step;

		float2 n = float2(0.0, step.y);
		float2 ne = float2(step.x, step.y);
		float2 e = float2(step.x, 0.0);
		float2 se = float2(step.x, -step.y);
		float2 s = float2(0.0, -step.y);
		float2 sw = float2(-step.x, -step.y);
		float2 w = float2(-step.x, 0.0);
		float2 nw = float2(-step.x, step.y);

		float4 u = tex2D(_bufferB, frac(aUv));
		float4 u_n = tex2D(_bufferB, frac(aUv + n));
		float4 u_e = tex2D(_bufferB, frac(aUv + e));
		float4 u_s = tex2D(_bufferB, frac(aUv + s));
		float4 u_w = tex2D(_bufferB, frac(aUv + w));
		float4 u_nw = tex2D(_bufferB, frac(aUv + nw));
		float4 u_sw = tex2D(_bufferB, frac(aUv + sw));
		float4 u_ne = tex2D(_bufferB, frac(aUv + ne));
		float4 u_se = tex2D(_bufferB, frac(aUv + se));

		return gaussian(u, u_nw, u_n, u_ne, u_w, u_e, u_sw, u_s, u_se);
	}

	#define SQRT_3_OVER_2 0.86602540378
	#define SQRT_3_OVER_2_INV 0.13397459621

	float2 diagH(float2 x, float2 x_v, float2 x_h, float2 x_d) {
		return 0.5 * ((x + x_v) * SQRT_3_OVER_2_INV + (x_h + x_d) * SQRT_3_OVER_2);
	}

	float2 diagV(float2 x, float2 x_v, float2 x_h, float2 x_d) {
		return 0.5 * ((x + x_h) * SQRT_3_OVER_2_INV + (x_v + x_d) * SQRT_3_OVER_2);
	}

	fixed4 frag(v2f_img i) : SV_Target
	{
		float2 vUv = i.uv;
		float2 texel = 1. / _ScreenParams.xy;

		float2 n = float2(0.0, 1.0);
		float2 ne = float2(1.0, 1.0);
		float2 e = float2(1.0, 0.0);
		float2 se = float2(1.0, -1.0);
		float2 s = float2(0.0, -1.0);
		float2 sw = float2(-1.0, -1.0);
		float2 w = float2(-1.0, 0.0);
		float2 nw = float2(-1.0, 1.0);

		float4 u = tex2D(_bufferB, frac(vUv));
		float4 u_n = tex2D(_bufferB, frac(vUv + texel*n));
		float4 u_e = tex2D(_bufferB, frac(vUv + texel*e));
		float4 u_s = tex2D(_bufferB, frac(vUv + texel*s));
		float4 u_w = tex2D(_bufferB, frac(vUv + texel*w));
		float4 u_nw = tex2D(_bufferB, frac(vUv + texel*nw));
		float4 u_sw = tex2D(_bufferB, frac(vUv + texel*sw));
		float4 u_ne = tex2D(_bufferB, frac(vUv + texel*ne));
		float4 u_se = tex2D(_bufferB, frac(vUv + texel*se));

		const float vx = 0.5;
		const float vy = SQRT_3_OVER_2;
		const float hx = SQRT_3_OVER_2;
		const float hy = 0.5;

		float di_n = nl(distance(u_n.xy + n, u.xy));
		float di_w = nl(distance(u_w.xy + w, u.xy));
		float di_e = nl(distance(u_e.xy + e, u.xy));
		float di_s = nl(distance(u_s.xy + s, u.xy));

		float di_nne = nl(distance((diagV(u.xy, u_n.xy, u_e.xy, u_ne.xy) + float2(+vx, +vy)), u.xy));
		float di_ene = nl(distance((diagH(u.xy, u_n.xy, u_e.xy, u_ne.xy) + float2(+hx, +hy)), u.xy));
		float di_ese = nl(distance((diagH(u.xy, u_s.xy, u_e.xy, u_se.xy) + float2(+hx, -hy)), u.xy));
		float di_sse = nl(distance((diagV(u.xy, u_s.xy, u_e.xy, u_se.xy) + float2(+vx, -vy)), u.xy));
		float di_ssw = nl(distance((diagV(u.xy, u_s.xy, u_w.xy, u_sw.xy) + float2(-vx, -vy)), u.xy));
		float di_wsw = nl(distance((diagH(u.xy, u_s.xy, u_w.xy, u_sw.xy) + float2(-hx, -hy)), u.xy));
		float di_wnw = nl(distance((diagH(u.xy, u_n.xy, u_w.xy, u_nw.xy) + float2(-hx, +hy)), u.xy));
		float di_nnw = nl(distance((diagV(u.xy, u_n.xy, u_w.xy, u_nw.xy) + float2(-vx, +vy)), u.xy));

		float2 xy_n = u_n.xy + n - u.xy;
		float2 xy_w = u_w.xy + w - u.xy;
		float2 xy_e = u_e.xy + e - u.xy;
		float2 xy_s = u_s.xy + s - u.xy;

		float2 xy_nne = (diagV(u.xy, u_n.xy, u_e.xy, u_ne.xy) + float2(+vx, +vy)) - u.xy;
		float2 xy_ene = (diagH(u.xy, u_n.xy, u_e.xy, u_ne.xy) + float2(+hx, +hy)) - u.xy;
		float2 xy_ese = (diagH(u.xy, u_s.xy, u_e.xy, u_se.xy) + float2(+hx, -hy)) - u.xy;
		float2 xy_sse = (diagV(u.xy, u_s.xy, u_e.xy, u_se.xy) + float2(+vx, -vy)) - u.xy;
		float2 xy_ssw = (diagV(u.xy, u_s.xy, u_w.xy, u_sw.xy) + float2(-vx, -vy)) - u.xy;
		float2 xy_wsw = (diagH(u.xy, u_s.xy, u_w.xy, u_sw.xy) + float2(-hx, -hy)) - u.xy;
		float2 xy_wnw = (diagH(u.xy, u_n.xy, u_w.xy, u_nw.xy) + float2(-hx, +hy)) - u.xy;
		float2 xy_nnw = (diagV(u.xy, u_n.xy, u_w.xy, u_nw.xy) + float2(-vx, +vy)) - u.xy;

		float2 ma = di_nne * xy_nne + di_ene * xy_ene + di_ese * xy_ese + di_sse * xy_sse + di_ssw * xy_ssw + di_wsw * xy_wsw + di_wnw * xy_wnw + di_nnw * xy_nnw + di_n * xy_n + di_w * xy_w + di_e * xy_e + di_s * xy_s;

		float4 u_blur = gaussian(u, u_nw, u_n, u_ne, u_w, u_e, u_sw, u_s, u_se);

		float4 au = advect(u.xy, vUv, texel);
		float4 av = advect(u.zw, vUv, texel);

		float2 dv = av.zw + TIMESTEP * ma;
		float2 du = au.xy + TIMESTEP * dv;

		/*
		if (iMouse.z > 0.0) {
		float2 d = fragCoord.xy - iMouse.xy;
		float m = exp(-length(d) / 50.0);
		du += 0.2 * m * normz(d);
		}
		*/
		// for now
		float4 init4 = tex2D(iChannel1, vUv);
		float2 init = init4.xy;
		// initialize with noise
		if ((length(u) < 0.001 && length(init) > 0.001) || reset()) {
			//fragColor = 8.0 * (float4(-0.5) + float4(init.xy, init.xy));
			return 8.0 * float4(-0.5, -0.5, -0.5, -0.5) + float4(init.xy, init.xy);
		}
		else {
			du = length(du) > 1.0 ? normz(du) : du;
			//fragColor = float4(du, dv);
			return float4(du, dv);
		}

	}
	ENDCG
	}
	GrabPass{ "_bufferA" }

	Pass{
		CGPROGRAM
	#pragma vertex vert_img
	#pragma fragment frag
	#include "UnityCG.cginc"

	sampler2D _bufferA;

	float2 normz(float2 x) {
		return x == float2(0.0, 0.0) ? float2(0.0, 0.0) : normalize(x);
	}

	// reverse advection
	float4 advect(float2 ab, float2 vUv, float2 step, float sc) {

		float2 aUv = vUv - ab * sc * step;

		static const float _G0 = 0.25; // center weight
		static const float _G1 = 0.125; // edge-neighbors
		static const float _G2 = 0.0625; // vertex-neighbors

										 // 3x3 neighborhood coordinates
		float step_x = step.x;
		float step_y = step.y;
		float2 n = float2(0.0, step_y);
		float2 ne = float2(step_x, step_y);
		float2 e = float2(step_x, 0.0);
		float2 se = float2(step_x, -step_y);
		float2 s = float2(0.0, -step_y);
		float2 sw = float2(-step_x, -step_y);
		float2 w = float2(-step_x, 0.0);
		float2 nw = float2(-step_x, step_y);

		float4 uv = tex2D(_bufferA, frac(aUv));
		float4 uv_n = tex2D(_bufferA, frac(aUv + n));
		float4 uv_e = tex2D(_bufferA, frac(aUv + e));
		float4 uv_s = tex2D(_bufferA, frac(aUv + s));
		float4 uv_w = tex2D(_bufferA, frac(aUv + w));
		float4 uv_nw = tex2D(_bufferA, frac(aUv + nw));
		float4 uv_sw = tex2D(_bufferA, frac(aUv + sw));
		float4 uv_ne = tex2D(_bufferA, frac(aUv + ne));
		float4 uv_se = tex2D(_bufferA, frac(aUv + se));

		return _G0*uv + _G1*(uv_n + uv_e + uv_w + uv_s) + _G2*(uv_nw + uv_sw + uv_ne + uv_se);
	}

	fixed4 frag(v2f_img i) : SV_Target
	{
		static const float _K0 = -20.0 / 6.0; // center weight
		static const float _K1 = 4.0 / 6.0;   // edge-neighbors
		static const float _K2 = 1.0 / 6.0;   // vertex-neighbors
		static const float cs = -3.0;  // curl scale
		static const float ls = 3.0;  // laplacian scale
		static const float ps = 0.0;  // laplacian of divergence scale
		static const float ds = -12.0; // divergence scale
		static const float dp = -6.0; // divergence update scale
		static const float pl = 0.3;   // divergence smoothing
		static const float ad = 6.0;   // advection distance scale
		static const float pwr = 1.0;  // power when deriving rotation angle from curl
		static const float amp = 1.0;  // self-amplification
		static const float upd = 0.99;  // update smoothing
		static const float sq2 = 0.6;  // diagonal weight

		float2 vUv = i.uv;
		float2 texel = 1. / _ScreenParams.xy;

		// 3x3 neighborhood coordinates
		float step_x = texel.x;
		float step_y = texel.y;
		float2 n = float2(0.0, step_y);
		float2 ne = float2(step_x, step_y);
		float2 e = float2(step_x, 0.0);
		float2 se = float2(step_x, -step_y);
		float2 s = float2(0.0, -step_y);
		float2 sw = float2(-step_x, -step_y);
		float2 w = float2(-step_x, 0.0);
		float2 nw = float2(-step_x, step_y);

		float4 uv = tex2D(_bufferA, frac(vUv));
		float4 uv_n = tex2D(_bufferA, frac(vUv + n));
		float4 uv_e = tex2D(_bufferA, frac(vUv + e));
		float4 uv_s = tex2D(_bufferA, frac(vUv + s));
		float4 uv_w = tex2D(_bufferA, frac(vUv + w));
		float4 uv_nw = tex2D(_bufferA, frac(vUv + nw));
		float4 uv_sw = tex2D(_bufferA, frac(vUv + sw));
		float4 uv_ne = tex2D(_bufferA, frac(vUv + ne));
		float4 uv_se = tex2D(_bufferA, frac(vUv + se));

		// uv.x and uv.y are the x and y components, uv.z and uv.w accumulate divergence

		// laplacian of all components
		float4 lapl = _K0*uv + _K1*(uv_n + uv_e + uv_w + uv_s) + _K2*(uv_nw + uv_sw + uv_ne + uv_se);

		// calculate curl
		// vectors point clockwise about the center point
		float curl = uv_n.x - uv_s.x - uv_e.y + uv_w.y + sq2 * (uv_nw.x + uv_nw.y + uv_ne.x - uv_ne.y + uv_sw.y - uv_sw.x - uv_se.y - uv_se.x);

		// compute angle of rotation from curl
		float sc = cs * sign(curl) * pow(abs(curl), pwr);

		// calculate divergence
		// vectors point inwards towards the center point
		float div = uv_s.y - uv_n.y - uv_e.x + uv_w.x + sq2 * (uv_nw.x - uv_nw.y - uv_ne.x - uv_ne.y + uv_sw.x + uv_sw.y + uv_se.y - uv_se.x);

		float2 norm = normz(uv.xy);

		float sdx = uv.z + dp * uv.x * div + pl * lapl.z;
		float sdy = uv.w + dp * uv.y * div + pl * lapl.w;

		float2 ab = advect(float2(uv.x, uv.y), vUv, texel, ad).xy;

		// temp values for the update rule
		float ta = amp * ab.x + ls * lapl.x + norm.x * ps * lapl.z + ds * sdx;
		float tb = amp * ab.y + ls * lapl.y + norm.y * ps * lapl.w + ds * sdy;

		// rotate
		float a = ta * cos(sc) - tb * sin(sc);
		float b = ta * sin(sc) + tb * cos(sc);

		float4 abd = upd * uv + (1.0 - upd) * float4(a, b, sdx, sdy);

		//fragColor = float4(abd);
		return float4(abd);

		//abd.xy = clamp(length(abd.xy) > 1.0 ? normz(abd.xy) : abd.xy, -1.0, 1.0);
		//fragColor = float4(abd);
	}
	ENDCG
	}
	GrabPass{ "_bufferB" }
	Pass{
	CGPROGRAM
	#pragma vertex vert_img
	#pragma fragment frag
	#include "UnityCG.cginc"

	// displacement amount
	#define DISP_SCALE 1.0

	// chromatic dispersion samples
	#define SAMPLES 64

	// contrast
	#define SIGMOID_CONTRAST 12.0

	// channels to use for displacement, either xy or zw
	#define CH xy

	sampler2D _bufferA;
	sampler2D iChannel2;

	float3 contrast(float3 x) {
		return 1.0 / (1.0 + exp(-SIGMOID_CONTRAST * (x - 0.5)));
	}

	float2 normz(float2 x) {
		return x == float2(0, 0) ? float2(0, 0) : normalize(x);
	}

	float3 sampleWeights(float i) {
		return float3(i * i, 46.6666*pow((1.0 - i)*i, 3.0), (1.0 - i) * (1.0 - i));
	}

	float3 sampleDisp(float2 uv, float2 dispNorm, float disp) {
		float3 col = float3(0, 0, 0);
		const float SD = 1.0 / float(SAMPLES);
		float wl = 0.0;
		float3 denom = float3(0, 0, 0);
		for (int i = 0; i < SAMPLES; i++) {
			float3 sw = sampleWeights(wl);
			denom += sw;
			col += sw * tex2D(iChannel2, uv + dispNorm * disp * wl).xyz;
			wl += SD;
		}

		// For a large enough number of samples,
		// the return below is equivalent to 3.0 * col * SD;
		return col / denom;
	}

	fixed4 frag(v2f_img i) : SV_Target
	{
		float2 texel = 1. / _ScreenParams.xy;
		float2 uv = i.uv;

		float2 n = float2(0.0, texel.y);
		float2 e = float2(texel.x, 0.0);
		float2 s = float2(0.0, -texel.y);
		float2 w = float2(-texel.x, 0.0);

		float2 d = tex2D(_bufferA, uv).CH;
		float2 d_n = tex2D(_bufferA, frac(uv + n)).CH;
		float2 d_e = tex2D(_bufferA, frac(uv + e)).CH;
		float2 d_s = tex2D(_bufferA, frac(uv + s)).CH;
		float2 d_w = tex2D(_bufferA, frac(uv + w)).CH;

		// antialias our vector field by blurring
		float2 db = 0.4 * d + 0.15 * (d_n + d_e + d_s + d_w);

		float ld = length(db);
		float2 ln = normz(db);

		float3 col = sampleDisp(uv, ln, DISP_SCALE * ld);

		//fragColor = vec4(contrast(col), 1.0);
		return float4(contrast(col), 1.0);
	}
	ENDCG
	}
	}
	FallBack "Diffuse"
}