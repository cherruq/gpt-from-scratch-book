'use strict';

(function () {
    function probabilities(scores, temperature) {
        const maximum = Math.max(...scores);
        const weights = scores.map(score => Math.exp((score - maximum) / temperature));
        const total = weights.reduce((a, b) => a + b, 0);
        return weights.map(weight => weight / total);
    }

    function element(name, text) {
        const node = document.createElement(name);
        if (text !== undefined) node.textContent = text;
        return node;
    }

    const svgNamespace = 'http://www.w3.org/2000/svg';
    function svgElement(name, attributes) {
        const node = document.createElementNS(svgNamespace, name);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        return node;
    }

    function svgChart(viewBox, label) {
        return svgElement('svg', {viewBox, role: 'img', 'aria-label': label});
    }

    function angleValues(degrees) {
        const radians = degrees * Math.PI / 180;
        return [Math.cos(radians), Math.sin(radians)];
    }

    function trajectory(rate, count) {
        return Array.from({length: count}, (_, i) => 2 * (1 - 2 * rate) ** i);
    }

    function renderMatrix(caption, rows, selected = () => false) {
        const table = element('table');
        table.className = 'demo-matrix';
        const title = element('caption', caption);
        const body = element('tbody');
        rows.forEach((row, i) => {
            const tr = element('tr');
            row.forEach((value, j) => {
                const td = element('td', String(value));
                if (selected(i, j)) td.className = 'selected';
                tr.append(td);
            });
            body.append(tr);
        });
        table.append(title, body);
        return table;
    }

    function mergePair(ids, pair, replacement) {
        const result = [];
        for (let i = 0; i < ids.length; i++) {
            if (ids[i] === pair[0] && ids[i + 1] === pair[1]) {
                result.push(replacement);
                i++;
            } else result.push(ids[i]);
        }
        return result;
    }

    function embeddingData(second) {
        const ids = [1, second, 0];
        const embeddings = [[1, 2], [3, 4], [5, 6]];
        const upstream = [[1, 1], [2, 3], [4, 5]];
        const gradient = [[0, 0], [0, 0], [0, 0]];
        ids.forEach((id, i) => upstream[i].forEach((value, j) => { gradient[id][j] += value; }));
        return {ids, embeddings, gradient};
    }

    function attentionData() {
        const q = [[1, 0], [0, 1]];
        const v = [[2, 0], [0, 4]];
        const scores = q.map(row => q.map(key => row.reduce((sum, x, d) => sum + x * key[d], 0) / Math.sqrt(2)));
        const weights = scores.map(row => probabilities(row, 1));
        const output = weights.map(row => v[0].map((_, d) =>
            row.reduce((sum, weight, j) => sum + weight * v[j][d], 0)));
        return {q, v, scores, weights, output};
    }

    function slider(host, title, min, max, step, initial, render) {
        const controls = element('div');
        controls.className = 'demo-controls';
        const label = element('label', title);
        const input = element('input');
        Object.assign(input, {type: 'range', min, max, step, value: initial});
        const reset = element('button', '重置');
        reset.type = 'button';
        const output = element('output');
        const view = element('div');
        label.append(input, output);
        controls.append(label, reset);
        host.append(controls, view);
        const update = () => {
            output.value = input.value;
            render(view, Number(input.value));
        };
        input.addEventListener('input', update);
        reset.addEventListener('click', () => { input.value = initial; update(); });
        update();
    }

    function bars(host, labels, values) {
        const list = element('ul');
        list.className = 'bars';
        labels.forEach((label, i) => {
            const item = element('li');
            const meter = element('meter');
            Object.assign(meter, {min: 0, max: 1, value: values[i]});
            meter.setAttribute('aria-label', label);
            item.append(element('span', label), meter, element('span', values[i].toFixed(4)));
            list.append(item);
        });
        host.replaceChildren(list);
    }

    const widgets = {
        temperature(host) {
            slider(host, '温度', 0.1, 3, 0.1, 1, (view, value) => {
                bars(view, ['春', '夏', '秋', '冬'], probabilities([0, 2, 1, -1], value));
            });
        },
        angle(host) {
            slider(host, '夹角（度）', 0, 180, 1, 60, (view, value) => {
                const [x, y] = angleValues(value);
                const svg = svgChart('0 0 240 134', '固定单位圆，一条向量水平、另一条向量旋转');
                svg.append(svgElement('circle', {cx: 120, cy: 105, r: 80, fill: 'none', stroke: 'var(--line)'}));
                svg.append(svgElement('line', {x1: 120, y1: 105, x2: 200, y2: 105,
                    stroke: 'var(--muted)', 'stroke-width': 3}));
                svg.append(svgElement('line', {x1: 120, y1: 105, x2: 120 + 80 * x,
                    y2: 105 - 80 * y, stroke: 'currentColor', 'stroke-width': 4}));
                view.replaceChildren(svg, element('p', '点积 = ' + x.toFixed(4)));
            });
        },
        descent(host) {
            slider(host, '学习率', 0, 1.15, 0.01, 0.1, (view, rate) => {
                const values = trajectory(rate, 9);
                const maximum = Math.max(...values.map(Math.abs), 2);
                const svg = svgChart('0 0 280 140', '横轴为更新次数，纵轴为参数 w');
                const points = values.map((w, i) => [12 + i * 32, 70 - w * 58 / maximum]);
                svg.append(svgElement('line', {x1: 0, y1: 70, x2: 280, y2: 70, stroke: 'var(--line)'}));
                svg.append(svgElement('polyline', {points: points.map(point => point.join(',')).join(' '),
                    fill: 'none', stroke: 'currentColor', 'stroke-width': 2}));
                for (const [cx, cy] of points) svg.append(svgElement('circle', {cx, cy, r: 3, fill: 'currentColor'}));
                view.replaceChildren(svg, element('p', values.map(w => w.toFixed(3)).join(' → ')));
            });
        },
        matrix(host) {
            slider(host, '第一行的首项', -3, 5, 1, 1, (view, a) => {
                view.replaceChildren(renderMatrix('X', [[a, 2], [3, 4]], i => i === 0),
                    renderMatrix('XW', [[2 * a + 2, 6, 4 - a], [10, 12, 5]], i => i === 0));
            });
        },
        chain(host) {
            slider(host, '输入 x', -3, 3, 1, 2, (view, x) => {
                const svg = svgChart('0 0 290 136', '平方和恒等两条分支共同传回 x');
                for (const [x1, y1, x2, y2] of [[50, 66, 128, 33], [50, 66, 128, 101],
                    [162, 33, 241, 66], [162, 101, 241, 66]]) {
                    svg.append(svgElement('line', {x1, y1, x2, y2, stroke: 'currentColor', 'stroke-width': 2}));
                }
                for (const [cx, cy, label] of [[35, 66, 'x'], [145, 33, 'x²'], [145, 101, 'x'], [256, 66, 'y']]) {
                    svg.append(svgElement('circle', {cx, cy, r: 17,
                        fill: 'var(--surface)', stroke: 'currentColor'}));
                    const caption = svgElement('text', {x: cx, y: cy + 5, 'text-anchor': 'middle', fill: 'currentColor'});
                    caption.textContent = label;
                    svg.append(caption);
                }
                view.replaceChildren(svg,
                    element('p', 'y = ' + (x * x + x) + ' · 梯度 = ' + (2 * x) + ' + 1 = ' + (2 * x + 1)));
            });
        },
        residual(host) {
            slider(host, '线性支路系数 a', -2, 3, 1, 2, (view, a) => {
                view.replaceChildren(element('p', '输入 2，输出 2 + ' + (a * 2) + ' = ' + (2 + a * 2)),
                    element('p', '梯度：直路 1 + 支路 ' + a + ' = ' + (1 + a)));
            });
        },
        bpe(host) {
            slider(host, '已应用的合并数', 0, 2, 1, 0, (view, step) => {
                const merges = [[[97, 98], 258], [[258, 258], 259]];
                let ids = [97, 98, 97, 98];
                for (const [pair, replacement] of merges.slice(0, step)) ids = mergePair(ids, pair, replacement);
                view.replaceChildren(element('p', 'Token ID：[' + ids.join(', ') + ']'));
            });
        },
        embedding(host) {
            slider(host, '第二个 Token ID', 0, 2, 1, 1, (view, second) => {
                const {ids, embeddings, gradient} = embeddingData(second);
                view.replaceChildren(renderMatrix('查表输出 · ID ' + ids.join(', '),
                    ids.map(id => embeddings[id]), i => i === 1),
                    renderMatrix('嵌入表梯度', gradient, i => i === second));
            });
        },
        attention(host) {
            const {q, v, scores, weights, output} = attentionData();
            const stages = [[['Q 与 K', q], ['V', v]], [['分数', scores]], [['权重', weights]], [['输出', output]]];
            slider(host, '计算阶段', 0, 3, 1, 0, (view, step) => {
                view.replaceChildren(...stages[step].map(([label, rows]) =>
                    renderMatrix(label, rows.map(row => row.map(x => x.toFixed(4))))));
            });
        },
        causal(host) {
            slider(host, '查询位置', 0, 3, 1, 1, (view, index) => {
                const matrix = Array.from({length: 4}, (_, i) =>
                    Array.from({length: 4}, (_, j) => j <= i ? '可读' : '屏蔽'));
                view.replaceChildren(renderMatrix('行是查询 · 列是键', matrix, (i, j) => i === index && j <= i));
            });
        },
        heads(host) {
            slider(host, '时间位置', 0, 3, 1, 2, (view, index) => {
                const rows = Array.from({length: 4}, (_, t) => [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3]);
                const row = rows[index];
                view.replaceChildren(renderMatrix('时间 × 通道', rows, i => i === index),
                    renderMatrix('所选位置 · 两个头', [row.slice(0, 2), row.slice(2)]));
            });
        },
        sampling(host) {
            slider(host, 'top-p 阈值', 0.05, 1, 0.05, 0.75, (view, threshold) => {
                const initial = [0.5, 0.3, 0.2];
                let cumulative = 0;
                const kept = initial.map(p => {
                    if (cumulative >= threshold) return 0;
                    cumulative += p;
                    return p;
                });
                bars(view, ['春', '夏', '秋'], kept.map(p => p / cumulative));
            });
        },
        loss(host) {
            const rows = (window.BOOK_RUN ?? []).filter(row => row.validation_nll !== undefined);
            if (!rows.length) return;
            slider(host, '记录点', 0, rows.length - 1, 1, 0, (view, index) => {
                const maximum = rows[0].validation_nll;
                const svg = svgChart('0 0 280 140', '横轴为训练步数，纵轴为验证损失');
                const points = rows.map(row => [12 + 256 * row.step / rows.at(-1).step,
                    128 - 116 * row.validation_nll / maximum]);
                svg.append(svgElement('polyline', {points: points.map(point => point.join(',')).join(' '),
                    fill: 'none', stroke: 'currentColor', 'stroke-width': 2}));
                points.forEach(([cx, cy], i) => svg.append(svgElement('circle',
                    {cx, cy, r: i === index ? 6 : 3, fill: 'currentColor'})));
                const row = rows[index];
                view.replaceChildren(svg, element('p', '步数 ' + row.step + ' · 验证 NLL ' +
                    row.validation_nll.toFixed(4) + ' · PPL ' + row.validation_ppl.toFixed(2)));
            });
        },
        cache(host) {
            slider(host, '缓存位置数', 1, 128, 1, 4, (view, length) => {
                const bytes = 2 * 2 * 2 * 2 * length * 2 * 8;
                view.replaceChildren(element('p', String(bytes) + ' 字节 · 两层、批量 2、2 个 KV 头、float64'));
            });
        },
        attribution(host) {
            const metrics = window.BOOK_INTERPRETATION;
            if (!metrics?.attention_row) return;
            const values = metrics.attention_row;
            const tokens = metrics.attention_tokens;
            slider(host, '键位置', 0, values.length - 1, 1, values.length - 1, (view, i) => {
                const line = element('p', '所选：' + tokens[i] + ' · 权重 ' + values[i].toFixed(4));
                view.replaceChildren(line);
                const chart = element('div');
                bars(chart, tokens, values);
                chart.querySelectorAll('li')[i].className = 'selected';
                view.append(chart);
            });
        },
    };

    function mount(figure) {
        const demo = widgets[figure.dataset.demo];
        if (demo) demo(figure.querySelector('.demo'));
    }

    if (typeof module !== 'undefined') {
        module.exports = {probabilities, angleValues, trajectory, mergePair, embeddingData, attentionData};
    }
    else window.BOOK_DEMOS = {mount};
})();
