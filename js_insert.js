
/* Page scripts
 *  
 */

function getRunner() {
    return Runner.instance_;
}

function update(r) {
    let obstacles = r.horizon.obstacles;
    let obstacle_positions = [];
    for (let o of obstacles) {
        obstacle_positions.push({
            x: o.xPos,
            y: o.yPos
        });
    }
    let trex_pos = {
        x: r.tRex.xPos,
        y: r.tRex.yPos
    };
    return {
        trex: trex_pos,
        obstacles: obstacle_positions
    }
}


/* SQLite Scripts
 * 
 * npm install sqlite3
 * 
 * const sqlite3 = require('sqlite3');
 * 
 * 
 */







/* Node Scripts
 *  
 */

// npm install selenium-webdriver
const selenium = require('selenium-webdriver');
const driver = selenium.Builder().forBriwser('chrome').build();
// const chrome = require('selenium-webdriver/chrome');


// get a page
// await driver.get('http://www.google.com/');

// 
// await driver.findElement(selenium.By.name('q')).sendKeys('webdriver',Key.RETURN);

// 
// await driver.wait(until.titleIs('webdriver - Google Search'),1000);










