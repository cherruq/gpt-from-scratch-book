'use strict';

function readSetting(key, fallback) {
    try {
        return JSON.parse(localStorage.getItem(key)) ?? fallback;
    } catch {
        return fallback;
    }
}

function saveSetting(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch {
        // Readers may block storage, particularly for local files.
    }
}

const root = document.documentElement;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
root.dataset.theme = readSetting('gpt-book-theme', dark ? 'dark' : 'light');
const theme = document.getElementById('theme');
function syncTheme() {
    theme.textContent = root.dataset.theme === 'dark' ? '浅色' : '深色';
}
syncTheme();
theme.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    saveSetting('gpt-book-theme', root.dataset.theme);
    syncTheme();
});
document.querySelector('.toolbar').hidden = false;

const side = document.querySelector('.library details');
if (matchMedia('(max-width: 700px)').matches) side.open = false;

const index = window.BOOK_INDEX ?? [];
const search = document.getElementById('search');
const query = document.getElementById('query');
const results = document.getElementById('search-results');
function searchBook() {
    const words = query.value.normalize('NFKC').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    results.replaceChildren();
    if (!words.length) return;
    const matches = index.map(page => {
        const title = page.title.normalize('NFKC').toLocaleLowerCase();
        const text = page.text.normalize('NFKC').toLocaleLowerCase();
        if (!words.every(word => title.includes(word) || text.includes(word))) return null;
        return {page, score: words.filter(word => title.includes(word)).length, offset: text.indexOf(words[0])};
    }).filter(Boolean).sort((a, b) => b.score - a.score || a.page.id - b.page.id);
    for (const {page, offset} of matches.slice(0, 20)) {
        const link = document.createElement('a');
        const snippet = document.createElement('small');
        link.href = page.href;
        link.textContent = page.id + ' · ' + page.title;
        snippet.textContent = page.text.slice(Math.max(0, offset - 22), Math.max(0, offset - 22) + 85);
        link.append(snippet);
        results.append(link);
    }
    if (!matches.length) results.textContent = '没有匹配的章节。';
}
document.getElementById('open-search').addEventListener('click', () => {
    search.showModal();
    query.focus();
});
query.addEventListener('input', searchBook);

const completed = new Set(readSetting('gpt-book-read', []));
function syncProgress() {
    for (const link of document.querySelectorAll('.library a')) {
        const page = Number.parseInt(link.getAttribute('href'), 10);
        link.classList.toggle('is-read', completed.has(page));
    }
}
syncProgress();
const checkbox = document.getElementById('completed');
if (checkbox) {
    const chapter = Number(document.body.dataset.chapter);
    checkbox.parentElement.hidden = false;
    checkbox.checked = completed.has(chapter);
    checkbox.addEventListener('change', () => {
        if (checkbox.checked) completed.add(chapter);
        else completed.delete(chapter);
        saveSetting('gpt-book-read', Array.from(completed));
        syncProgress();
    });
}

for (const pre of document.querySelectorAll('main pre')) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'code-copy';
    button.textContent = '复制代码';
    pre.after(button);
    button.addEventListener('click', async () => {
        const code = pre.querySelector('code') || pre;
        try {
            if (!navigator.clipboard) throw new Error('No clipboard API');
            await navigator.clipboard.writeText(code.textContent.replace(/^\\n/, ''));
            button.textContent = '已复制';
        } catch {
            const range = document.createRange();
            range.selectNodeContents(code);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            button.textContent = document.execCommand('copy') ? '已复制' : '已选中，请手动复制';
        }
        setTimeout(() => { button.textContent = '复制代码'; }, 1800);
    });
}
for (const table of document.querySelectorAll('.prose table')) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    wrapper.tabIndex = 0;
    table.before(wrapper);
    wrapper.append(table);
}
for (const experiment of document.querySelectorAll('[data-demo]')) {
    window.BOOK_DEMOS?.mount(experiment);
}
