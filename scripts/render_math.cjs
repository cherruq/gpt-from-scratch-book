const katex = require('katex');
const fs = require('node:fs');

const equations = JSON.parse(fs.readFileSync(0, 'utf8'));
const rendered = equations.map(({tex, display, source}) => {
    try {
        return katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: 'error'});
    } catch (error) {
        throw new Error(source + ': ' + error.message);
    }
});
process.stdout.write(JSON.stringify(rendered));
