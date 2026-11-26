/* Run the generated book offline in a real Chromium browser. */
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const path = require('node:path');
const {chromium} = require('playwright');

async function main() {
    const executablePath = process.env.BOOK_CHROME || undefined;
    const browser = await chromium.launch({executablePath});
    const site = pathToFileURL(path.resolve(__dirname, '../dist') + path.sep);
    const errors = [];
    const reports = {};
    function track(page) {
        page.on('pageerror', error => errors.push(error.message));
        page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
        page.on('request', request => {
            if (/^https?:/.test(request.url())) errors.push('Remote resource ' + request.url());
        });
    }
    try {
        const view = await browser.newContext({offline: true, viewport: {width: 1440, height: 920},
            permissions: ['clipboard-read', 'clipboard-write']});
        const page = await view.newPage();
        track(page);
        const open = name => page.goto(new URL(name, site).href);
        await open('index.html');
        await page.getByRole('heading', {name: /从零实现 GPT/}).waitFor();
        await page.getByRole('button', {name: '搜索', exact: true}).click();
        await page.getByRole('searchbox').fill('自动微分引擎');
        await page.locator('#search-results a[href="15.html"]').click();
        await page.getByRole('heading', {name: '自动微分引擎', exact: true}).waitFor();
        assert((await page.locator('.katex').count()) > 0, 'Mathematics must be pre-rendered');
        assert(await page.evaluate(async () => {
            await document.fonts.ready;
            return Array.from(document.fonts).some(font =>
                font.family.includes('KaTeX') && font.status === 'loaded');
        }), 'At least one bundled KaTeX font must load');
        await page.locator('summary', {hasText: '参考答案'}).click();
        assert(await page.locator('.prose details').filter({hasText: '参考答案'}).evaluate(details => details.open));
        await page.locator('button.code-copy').first().click();
        await page.getByRole('button', {name: '已复制', exact: true}).waitFor({timeout: 3000});
        assert.match(await page.evaluate(() => navigator.clipboard.readText()), /class Value/);
        await page.locator('#theme').click();
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
        await page.locator('#completed').check();
        await page.reload();
        assert(await page.locator('#completed').isChecked(), 'Progress must survive reload');
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
        await open('index.html');
        assert.equal(await page.locator('.library a[href="15.html"].is-read').count(), 1);
        let checked = 0;
        const pages = await page.evaluate(() => window.BOOK_INDEX.map(entry => entry.href));
        for (const name of pages) {
            await open(name);
            for (const widget of await page.locator('figure[data-demo]').all()) {
                const slider = widget.getByRole('slider');
                await slider.waitFor();
                const initial = await slider.inputValue();
                const limit = await slider.getAttribute('max');
                await slider.focus();
                await slider.press(limit === initial ? 'Home' : 'End');
                assert.notEqual(await slider.inputValue(), initial);
                await widget.getByRole('button', {name: '重置'}).click();
                assert.equal(await slider.inputValue(), initial);
                checked++;
            }
        }
        reports.desktopWidgets = checked;
        await open('13.html');
        const initializationTable = page.getByRole('table').filter({hasText: '零初始化'});
        assert.equal(await initializationTable.getByRole('row').count(), 5);
        await open('17.html');
        const mergeTable = page.getByRole('table').filter({hasText: '实际词表大小'});
        assert.equal(await mergeTable.getByRole('row').count(), 4);
        await open('18.html');
        const embeddingTable = page.getByRole('table').filter({hasText: '反向加到该 ID'});
        assert.equal(await embeddingTable.getByRole('row').filter({hasText: '259'}).count(), 2);
        await open('19.html');
        const contextTable = page.getByRole('table').filter({hasText: '半径 1 的非零格'});
        assert.equal(await contextTable.getByRole('row').count(), 6);
        await open('20.html');
        const nextTokenTable = page.getByRole('table').filter({hasText: '平滑后概率'});
        assert.equal(await nextTokenTable.getByRole('cell', {name: '1/460', exact: true}).count(), 1);
        await open('21.html');
        const positionTable = page.getByRole('table').filter({hasText: '使用位置项'});
        assert.equal(await positionTable.getByRole('row').count(), 5);
        await open('22.html');
        const retrievalTable = page.getByRole('table').filter({hasText: '值的加权输出'});
        assert.equal(await retrievalTable.getByRole('row').count(), 6);
        await open('33.html');
        const precisionTable = page.getByRole('table').filter({hasText: 'bfloat16 autocast'});
        assert.equal(await precisionTable.getByRole('row').count(), 5);
        await open('35.html');
        const tailTable = page.getByRole('table').filter({hasText: 'Decode p99（ms）'});
        assert.equal(await tailTable.getByRole('row').count(), 7);
        await open('36.html');
        const groupedTable = page.getByRole('table').filter({hasText: '单块投影权重'});
        assert.equal(await groupedTable.getByRole('row').count(), 4);
        await open('37.html');
        const onlineTable = page.getByRole('table').filter({hasText: '已处理的键'});
        assert.equal(await onlineTable.getByRole('row').count(), 3);
        await open('38.html');
        const objectiveTable = page.getByRole('table').filter({hasText: '实际最大化目标'});
        assert.equal(await objectiveTable.getByRole('row').count(), 5);
        await open('39.html');
        const controlTable = page.getByRole('table').filter({hasText: '任意标签中位数'});
        assert.equal(await controlTable.getByRole('row').filter({hasText: '新模板合计'}).count(), 2);
        await open('40.html');
        const capstoneTable = page.getByRole('table').filter({hasText: '最终全批 NLL'});
        assert.equal(await capstoneTable.getByRole('row').count(), 3);
        reports.checkedExperimentTables = 14;
        const phone = await browser.newContext({offline: true, viewport: {width: 390, height: 844},
            isMobile: true, hasTouch: true});
        const mobile = await phone.newPage();
        track(mobile);
        const mobilePages = ['index.html', ...pages];
        for (const name of mobilePages) {
            await mobile.goto(new URL(name, site).href);
            await mobile.getByRole('heading', {level: 1}).waitFor();
            assert(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
                'Unexpected viewport overflow on ' + name);
        }
        assert(!(await mobile.locator('.library details').evaluate(details => details.open)));
        await mobile.locator('.library summary').click();
        assert(await mobile.locator('.library details').evaluate(details => details.open));
        await mobile.emulateMedia({media: 'print'});
        assert.equal(await mobile.locator('.top').evaluate(node => getComputedStyle(node).display), 'none');
        reports.mobilePages = mobilePages.length;
        reports.chrome = browser.version();
        assert.equal(errors.length, 0, errors.join('\n'));
        console.log(JSON.stringify(reports, null, 2));
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
